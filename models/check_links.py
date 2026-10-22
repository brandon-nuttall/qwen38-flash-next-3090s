#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright 2026 halt95 (Merlin modifications)

"""Link checker for the model recipes (models/), the release tooling (release/) and the container recipe (container/).

Standard library only; no network access. It checks, for every Markdown file under the given roots:

  * links  [text](target) and images ![alt](target), plus reference definitions  [id]: target
      - a relative target must exist (case-sensitive, relative to the file that links to it);
      - a '#fragment' on a Markdown target (or on the file itself) must match a heading, using GitHub's slug rules;
      - http(s)://, mailto: and other URL schemes are not fetched, but a github.com link into this repository's own old
        layout (blob/main/...) is reported, because it breaks when main changes layout.
  * path references: a `backticked` token that looks like a repository path (scripts/serve.sh, release/build-wheel.sh,
    models/flash-next/serve.sh, ...) must resolve against the file's directory, its recipe directory
    (models/<model>) or the repository root. Tokens naming files that were retired from the tree on purpose (the 0.30
    package route: upstream/PIN*, release/v*/..., patches/..., SHA256SUMS.v*) are listed once as "retired"
    rather than as errors, and a token that no longer resolves but whose file name exists elsewhere in the tree is listed as "moved".
    Path references are advisory (warnings) unless --strict-paths is given. CHANGELOG.md files and anything under benchmarks/ are
    historical records: their links are checked, their path references are not (they keep the paths of their time).

Usage (from the repository root):

    python models/check_links.py                       # models/ release/ container/
    python models/check_links.py models/qwen38-27b     # one recipe
    python models/check_links.py --strict-paths        # unresolved path references fail too
    python models/check_links.py --list-retired        # also print the retired-path mentions

Exit status: 0 when no link error (and, with --strict-paths, no unresolved path reference); 1 otherwise.
"""
from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
DEFAULT_ROOTS = ("models", "release", "container")

# Repository-wide documentation may be assembled separately from the recipes.
# A path reference to these root files is not an error.
PLANNED_ROOT_FILES = {"README.md", "NOTICE", "CHANGELOG.md", "SECURITY.md", "LICENSE"}

# Path tokens that name files of the retired 0.30 package route (kept at the old repositories' tags, not in this tree).
RETIRED = [
    re.compile(p)
    for p in (
        r"^upstream/",
        r"^release/v\d",
        r"^patches/",
        r"(^|/)SHA256SUMS(\.v[\d.]+)?$",
        r"(^|/)PACKAGE-MANIFEST\.md$",
        r"(^|/)build-v2(\.\d+)?\.sh$",
        r"(^|/)serve-v2(\.\d+)?\.sh$",
        r"(^|/)PIN(-v[\d.]+)?$",
        r"\.bundle(\.gz)?$",
        r"-combined\.diff$",
        r"^benchmarks/20\d\d-\d\d-\d\d/",
        r"^src-ref",
        r"^engine/",
        r"^quant/",  # Swift: the engine-clone layout of the retired package
        r"^vllm/_version\.py$",  # a build product
        r"^benchmarks/20\d\d-\d\d-\d\d$",
    )
]

PATH_TOP = (
    "models", "release", "container", "scripts", "tools", "docs", "tests", "benchmarks", "scales", "assets", "calib", "lxc",
    "licenses", "serve", "quant", "templates", "images", "upstream", "patches", "csrc", "vllm", "examples",
)
FILE_EXT = r"(?:sh|py|md|json|txt|pt|png|patch|list|sha256|jinja|yml|yaml|service|diff|cfg|toml)"
PATH_TOKEN = re.compile(
    r"^(?:\./)?(?:(?:%s)/[\w.@+\-/]+|[\w.\-]+/[\w.@+\-/]*[\w\-]\.%s)$" % ("|".join(PATH_TOP), FILE_EXT)
)

LINK = re.compile(r"(!?)\[((?:[^\[\]]|\[[^\]]*\])*)\]\(\s*(<[^>]*>|[^)\s]*)(?:\s+(?:\"[^\"]*\"|'[^']*'))?\s*\)")
REFDEF = re.compile(r"^\s{0,3}\[([^\]]+)\]:\s*(\S+)", re.M)
HEADING = re.compile(r"^(#{1,6})\s+(.*?)\s*#*\s*$")
INLINE_CODE = re.compile(r"(?<!`)(`+)(?!`)(.+?)(?<!`)\1(?!`)")
FENCE = re.compile(r"^\s*(```+|~~~+)")
SCHEME = re.compile(r"^[a-zA-Z][a-zA-Z0-9+.\-]*:")


def slugify(text: str) -> str:
    text = re.sub(r"!?\[([^\]]*)\]\([^)]*\)", r"\1", text)      # links/images -> their text
    text = re.sub(r"<[^>]+>", "", text)                          # inline html
    text = text.replace("`", "")
    text = text.strip().lower()
    out = []
    for ch in text:
        if ch.isalnum() or ch in "_-":
            out.append(ch)
        elif ch == " ":
            out.append("-")
        # everything else (punctuation, emphasis markers, symbols) is dropped, as GitHub does
    return "".join(out)


def anchors_of(path: Path, cache: dict) -> set:
    if path in cache:
        return cache[path]
    seen: dict = {}
    result = set()
    in_fence = False
    for line in path.read_text(encoding="utf-8", errors="replace").splitlines():
        if FENCE.match(line):
            in_fence = not in_fence
            continue
        if in_fence:
            continue
        m = HEADING.match(line)
        if m:
            s = slugify(m.group(2))
            n = seen.get(s, 0)
            seen[s] = n + 1
            result.add(s if n == 0 else f"{s}-{n}")
    cache[path] = result
    return result


def exists_exact(p: Path) -> bool:
    """Exists with the exact spelling of every path component (a case-insensitive checkout would hide a wrong-case link)."""
    p = Path(os.path.normpath(str(p)))
    if not p.exists():
        return False
    cur = p
    while cur != cur.parent:
        try:
            if cur.name not in os.listdir(cur.parent):
                return False
        except OSError:
            return False
        cur = cur.parent
    return True


_BASENAMES: set = set()


def basenames() -> set:
    if not _BASENAMES:
        for root in DEFAULT_ROOTS + ("scripts",):
            for p in (REPO / root).rglob("*"):
                if p.is_file():
                    _BASENAMES.add(p.name)
    return _BASENAMES


def md_files(roots):
    for root in roots:
        p = (REPO / root) if not Path(root).is_absolute() else Path(root)
        if p.is_file():
            yield p
        elif p.is_dir():
            yield from sorted(p.rglob("*.md"))


def strip_code(lines):
    """Yield (lineno, text, in_fence) with inline code spans blanked for the link scan."""
    in_fence = False
    for i, line in enumerate(lines, 1):
        if FENCE.match(line):
            in_fence = not in_fence
            yield i, line, True
            continue
        yield i, line, in_fence


def recipe_dir(path: Path):
    rel = path.relative_to(REPO).parts
    if len(rel) >= 2 and rel[0] == "models" and (REPO / "models" / rel[1]).is_dir():
        return REPO / "models" / rel[1]
    return None


def check_file(path: Path, cache: dict, errors: list, warns: list, retired: list):
    text = path.read_text(encoding="utf-8", errors="replace")
    lines = text.splitlines()
    rel = path.relative_to(REPO).as_posix()
    rdir = recipe_dir(path)
    historical = path.name == "CHANGELOG.md" or "benchmarks" in path.relative_to(REPO).parts
    for lineno, line, in_fence in strip_code(lines):
        if in_fence:
            continue
        scan = INLINE_CODE.sub(lambda m: " " * len(m.group(0)), line)
        targets = [m.group(3) for m in LINK.finditer(scan)]
        if REFDEF.match(scan):
            targets.append(REFDEF.match(scan).group(2))
        for t in targets:
            t = t.strip("<>")
            if not t:
                continue
            if SCHEME.match(t):
                if re.match(r"https?://github\.com/halt95/(merlin-engine|qwen38-flash-next-3090s)/(blob|tree|raw)/main/", t):
                    warns.append((rel, lineno, f"link into the old layout via 'main': {t}"))
                continue
            file_part, _, frag = t.partition("#")
            if file_part.startswith("/"):
                target = REPO / file_part.lstrip("/")
            elif file_part == "":
                target = path
            else:
                target = path.parent / file_part
            if not exists_exact(target):
                errors.append((rel, lineno, f"missing link target: {t}"))
                continue
            if frag and target.suffix.lower() == ".md" and target.is_file():
                if frag.lower() not in anchors_of(target, cache):
                    errors.append((rel, lineno, f"no such heading anchor: {t}"))
        # path references in `code` spans (not in historical records, which keep the paths of their time)
        if historical:
            continue
        for m in INLINE_CODE.finditer(line):
            tok = m.group(2).strip()
            if " " in tok or not PATH_TOKEN.match(tok):
                continue
            tok = tok[2:] if tok.startswith("./") else tok
            tok = tok.rstrip("/")
            cands = [path.parent / tok, REPO / tok]
            if rdir:
                cands.append(rdir / tok)
            if any(exists_exact(c) for c in cands):
                continue
            if tok in PLANNED_ROOT_FILES:
                continue
            if any(r.search(tok) for r in RETIRED):
                retired.append((rel, lineno, tok))
                continue
            if Path(tok).name in basenames():
                retired.append((rel, lineno, tok + "  (moved: same file name exists elsewhere in the tree)"))
                continue
            warns.append((rel, lineno, f"unresolved path reference `{tok}`"))


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("roots", nargs="*", default=list(DEFAULT_ROOTS))
    ap.add_argument("--strict-paths", action="store_true", help="unresolved path references are errors")
    ap.add_argument("--list-retired", action="store_true", help="print mentions of retired package-route files")
    args = ap.parse_args(argv)
    cache: dict = {}
    errors: list = []
    warns: list = []
    retired: list = []
    n = 0
    for f in md_files(args.roots):
        n += 1
        check_file(f, cache, errors, warns, retired)
    for rel, ln, msg in errors:
        print(f"ERROR {rel}:{ln}: {msg}")
    for rel, ln, msg in warns:
        print(f"{'ERROR' if args.strict_paths and msg.startswith('unresolved') else 'WARN '} {rel}:{ln}: {msg}")
    if args.list_retired:
        for rel, ln, tok in retired:
            print(f"RETIRED {rel}:{ln}: {tok}")
    print(f"checked {n} Markdown files: {len(errors)} link errors, {len(warns)} warnings, {len(retired)} mentions of retired package files")
    bad = len(errors) + (sum(1 for w in warns if w[2].startswith("unresolved")) if args.strict_paths else 0)
    return 1 if bad else 0


if __name__ == "__main__":
    sys.exit(main())
