#!/usr/bin/env bash
# Apply the FlashInfer 0.7.0.post1 sm_86 FP8 small-q verify tile to an existing venv.
# Installs the release's version-specific tile patch without enabling it.
# The launcher must set FLASHINFER_SM8X_FP8_SMALLQ=cta32 to select the 32-row tile.
#
#   release/apply-flashinfer-patch.sh <venv-path>
#
# Fails closed unless: flashinfer-python is exactly 0.7.0.post1 and installed in the venv's
# purelib; flashinfer-jit-cache is NOT installed (its prebuilt kernels would stand in for the
# patched JIT build); all five touched files are byte-identical to the 0.7.0.post1 wheel on PyPI (whole-file
# sha256) or byte-identical to the result of this patch. A tree that already carries exactly this patch
# (hashes, marker _sm8x_smallq_070_v1, patch cleanly reversible) is left alone, exit 0. The five files are
# checked before anything is changed and again after the patch is applied.
# Anything else (including the rev-1 _sm8x_smallq_v5 port, a half-applied or edited tree) is an error:
# reinstall flashinfer-python and run again. The package Python actually imports must be the one
# patched: `import flashinfer` in the venv's python, with a clean environment (`env -i`, `-I`: no
# PYTHONPATH, no user site, no working directory on sys.path) and with the caller's environment
# (PYTHONPATH and user site kept, working directory not added), must resolve to <purelib>/flashinfer.
# Concurrent runs on one venv are serialised (flock on <purelib>/.flashinfer-sm8x-patch.lock around
# the whole check / apply / post-check). Needs GNU patch and flock (util-linux).
set -euo pipefail

VENV="${1:?usage: apply-flashinfer-patch.sh <venv-path>}"
RELEASE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PATCH="$RELEASE/patches/flashinfer-0.7.0.post1/0001-sm8x-fp8-smallq-verify-tile-070.patch"
WANT_VERSION="0.7.0.post1"
MARKER="_sm8x_smallq_070_v1"
# every file the patch touches, relative to the flashinfer package directory
TARGETS=(
  data/csrc/batch_prefill.cu
  data/csrc/batch_prefill_paged_kernel_inst.jinja
  data/include/flashinfer/attention/prefill.cuh
  data/include/flashinfer/attention/scheduler.cuh
  jit/attention/modules.py
)
# sha256 of every touched file, in TARGETS order: PRISTINE_SHA = the files in flashinfer-python 0.7.0.post1 from PyPI
# (wheel sha256 c7adf826568d61fc1b7d3aadd4cae387a35a138bfc08e2deca2f34ecfa280716);
# PATCHED_SHA = the same files after `patch -p1 --fuzz=0` with this release's patch.
PRISTINE_SHA=(
  4190105e8b5c93621af44356b9aa50861560a5fbf8ff3279301487860604d932
  1e41a0ea9795847272674c1058a9d72f220facf296037cbf1f95ce6b77225a63
  e66bce2652c0a3c5b54f88510aff2b4b48ff6a52c4ee23fa97f91fa06a935b87
  425ba78815ba77ae848c67507d1af6b024d05edd257a7d9b331166afcef9b17c
  747c92bd33ecc40297bddfd4b9d388e86e1e70aac5d2d71a37bdfa752c19cca6
)
PATCHED_SHA=(
  0e669eee017eb01ef5c4b23915a026c9ad4217a5edce0c99be1926a58b3824c8
  500fcecb03e267e6c7bd85c1240ba34581e5e1f9b1c82dd671f1a669f740e041
  49cf57d1718eefe01d164a91ce36dd6bcbdf1b831d175fe5723b62a8dcc3ef61
  a8a7ccf79e488e7ff41dca6512f4c9bf6cb075ca4dfed0c9fb392de61eef80a4
  c38076b0c3c8a1d0e802dc6fc02a68e6da36604210bb03110df8774e220f37b5
)

die() { echo "apply-flashinfer-patch: FAIL: $*" >&2; exit 1; }

PY="$VENV/bin/python"
[ -x "$PY" ] || die "no python at $PY (expected a venv)"
[ -f "$PATCH" ] || die "missing $PATCH"
command -v patch >/dev/null 2>&1 || die "GNU patch is not installed (apt-get install patch)"
command -v flock >/dev/null 2>&1 || die "flock is not installed (apt-get install util-linux)"

SP="$(env -i "$PY" -I -c 'import sysconfig; print(sysconfig.get_paths()["purelib"])')"
FI="$SP/flashinfer"
[ -d "$FI" ] || die "flashinfer is not installed in $SP"

# The package `import flashinfer` resolves to must be $FI: first with a clean environment (env -i,
# isolated mode -I: no PYTHONPATH, no user site, no working directory), then with the caller's
# environment (PYTHONPATH and user site kept; -P keeps the working directory off sys.path, as for the
# vllm entry point). A shadowing copy on either path would be imported instead of the patched files.
import_origin() {
  "$@" - <<'PY'
import importlib.util, os
s = importlib.util.find_spec("flashinfer")
print(os.path.realpath(s.origin) if s is not None and s.origin else "<not importable>")
PY
}
WANT_ORIGIN="$(realpath "$FI/__init__.py")"
GOT="$(import_origin env -i "$PY" -I)" || die "import check failed (clean environment)"
[ "$GOT" = "$WANT_ORIGIN" ] \
  || die "with a clean environment, import flashinfer resolves to $GOT, not $WANT_ORIGIN"
GOT="$(import_origin "$PY" -P)" || die "import check failed (caller environment)"
[ "$GOT" = "$WANT_ORIGIN" ] \
  || die "with this environment (PYTHONPATH='${PYTHONPATH:-}'), import flashinfer resolves to $GOT, not $WANT_ORIGIN: remove the shadowing copy from the path"

# version + location of the installed distribution, and no flashinfer-jit-cache anywhere on the
# venv's path (import name or distribution)
"$PY" - "$SP" "$WANT_VERSION" <<'PY' || die "venv check failed (see above)"
import importlib.metadata as md, importlib.util, os, sys
sp, want = sys.argv[1], sys.argv[2]
bad = []
try:
    d = md.distribution("flashinfer-python")
except md.PackageNotFoundError:
    sys.exit("flashinfer-python is not installed")
if d.version != want:
    bad.append(f"flashinfer-python is {d.version}, this patch is for {want} only")
loc = os.path.realpath(str(d.locate_file("flashinfer")))
if loc != os.path.realpath(os.path.join(sp, "flashinfer")):
    bad.append(f"flashinfer-python resolves to {loc}, not {sp}/flashinfer")
if importlib.util.find_spec("flashinfer_jit_cache") is not None:
    bad.append("flashinfer_jit_cache is importable: uninstall flashinfer-jit-cache, its "
               "prebuilt kernels would shadow the patched JIT build")
for name in ("flashinfer-jit-cache", "flashinfer_jit_cache"):
    try:
        md.distribution(name)
        bad.append(f"distribution {name} is installed: uninstall it")
        break
    except md.PackageNotFoundError:
        pass
if bad:
    sys.exit("\n".join(bad))
PY
shopt -s nullglob
JC=("$SP"/flashinfer_jit_cache*)
shopt -u nullglob
[ "${#JC[@]}" -eq 0 ] || die "flashinfer_jit_cache files present in $SP: ${JC[*]}"
grep -q "__version__ = \"$WANT_VERSION\"" "$FI/_build_meta.py" \
  || die "$FI/_build_meta.py does not say $WANT_VERSION"
for f in "${TARGETS[@]}"; do [ -f "$FI/$f" ] || die "missing $FI/$f"; done

PATCH_OPTS=(-p1 --batch --fuzz=0 --no-backup-if-mismatch -d "$FI" -i "$PATCH")
# Forward runs never let GNU patch treat an already-applied hunk as a reversed patch and undo it.
FWD_OPTS=(--forward "${PATCH_OPTS[@]}")

# post-asserts: the applied tree carries every part of the patch
check_applied() {
  local n
  n="$(grep -c Sm8xApplySmallQ "$FI/data/include/flashinfer/attention/scheduler.cuh" || true)"
  [ "$n" = 3 ] || { echo "scheduler.cuh: Sm8xApplySmallQ x$n, want 3" >&2; return 1; }
  n="$(grep -c Sm8xResolveSmallQMode "$FI/data/include/flashinfer/attention/scheduler.cuh" || true)"
  [ "$n" = 3 ] || { echo "scheduler.cuh: Sm8xResolveSmallQMode x$n, want 3" >&2; return 1; }
  n="$(grep -c kSm8xKVTagE4M3 "$FI/data/csrc/batch_prefill.cu" || true)"
  [ "$n" = 2 ] || { echo "batch_prefill.cu: kSm8xKVTagE4M3 x$n, want 2" >&2; return 1; }
  n="$(grep -c kFp8SmallQ2x2 "$FI/data/include/flashinfer/attention/prefill.cuh" || true)"
  [ "$n" = 17 ] || { echo "prefill.cuh: kFp8SmallQ2x2 x$n, want 17" >&2; return 1; }
  n="$(grep -c "DTypeQKAccum(-5e4f)" "$FI/data/include/flashinfer/attention/prefill.cuh" || true)"
  [ "$n" = 2 ] || { echo "prefill.cuh: the 0.7.0 finite tile init x$n, want 2" >&2; return 1; }
  grep -q '\[16, 32, 64, 128\]' "$FI/data/csrc/batch_prefill_paged_kernel_inst.jinja" \
    || { echo "batch_prefill_paged_kernel_inst.jinja: no CTA32 instantiation at hd 256" >&2; return 1; }
  n="$(grep -c "$MARKER" "$FI/jit/attention/modules.py" || true)"
  [ "$n" = 1 ] || { echo "jit/attention/modules.py: $MARKER x$n, want 1" >&2; return 1; }
  "$PY" -c 'import ast, sys; ast.parse(open(sys.argv[1], encoding="utf-8").read())' \
    "$FI/jit/attention/modules.py" || { echo "jit/attention/modules.py does not parse" >&2; return 1; }
  # exactly this patch: it must reverse cleanly
  patch -R --dry-run "${PATCH_OPTS[@]}" >/dev/null \
    || { echo "the tree does not reverse cleanly with this patch" >&2; return 1; }
}

file_sha() { sha256sum "$1" | cut -d' ' -f1; }

# Classify the five touched files by whole-file sha256: all pristine, all exactly patched, or anything else.
# Sets TREE_STATE (pristine | patched | bad) and prints one line per file that is not in the expected state.
classify_tree() {
  local i h n_p=0 n_a=0 n_bad=0
  for i in "${!TARGETS[@]}"; do
    h="$(file_sha "$FI/${TARGETS[$i]}")"
    if [ "$h" = "${PRISTINE_SHA[$i]}" ]; then n_p=$((n_p + 1))
    elif [ "$h" = "${PATCHED_SHA[$i]}" ]; then n_a=$((n_a + 1))
    else n_bad=$((n_bad + 1)); echo "  ${TARGETS[$i]}: sha256 $h is neither pristine $WANT_VERSION nor exactly patched" >&2
    fi
  done
  if [ "$n_bad" -gt 0 ]; then TREE_STATE=bad
  elif [ "$n_p" = "${#TARGETS[@]}" ]; then TREE_STATE=pristine
  elif [ "$n_a" = "${#TARGETS[@]}" ]; then TREE_STATE=patched
  else TREE_STATE=bad; echo "  mixed state: $n_p file(s) pristine, $n_a patched (a partial apply)" >&2
  fi
}

# One transaction per venv: classify, apply and post-check under an exclusive lock, so a concurrent
# run can neither see a half-applied tree nor reverse a finished apply.
LOCK="$SP/.flashinfer-sm8x-patch.lock"
exec 9>"$LOCK" || die "cannot open lock file $LOCK"
flock -w 600 9 || die "timed out waiting for $LOCK (another apply-flashinfer-patch.sh on this venv?)"

# All five files are checked before anything is changed.
classify_tree
case "$TREE_STATE" in
  patched)
    check_applied || die "$FI hashes as patched but the post-apply checks fail; reinstall flashinfer-python $WANT_VERSION"
    echo "apply-flashinfer-patch: already applied ($MARKER) in $FI, nothing to do"
    exit 0 ;;
  pristine) ;;
  *) die "$FI is not pristine flashinfer-python $WANT_VERSION and not exactly this patch (edited, another revision, or a partial apply); reinstall flashinfer-python $WANT_VERSION" ;;
esac

patch --dry-run "${FWD_OPTS[@]}" >/dev/null \
  || die "patch does not apply cleanly to $FI (dry run); the tree is not pristine $WANT_VERSION"
patch "${FWD_OPTS[@]}" || die "patch failed in $FI after a clean dry run"

# drop bytecode compiled before the patch so the old jit/attention/modules.py is never imported
find "$FI" -name '__pycache__' -type d -prune -exec rm -rf {} +

# the applied tree must be byte-identical to the expected patched files
classify_tree
[ "$TREE_STATE" = patched ] || die "after applying, the files do not hash as the expected patched $WANT_VERSION tree"
check_applied || die "post-apply check failed in $FI"
echo "apply-flashinfer-patch: applied $(basename "$PATCH") in $FI"
