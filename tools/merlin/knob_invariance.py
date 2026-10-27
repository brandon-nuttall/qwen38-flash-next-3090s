#!/usr/bin/env python3
"""Invariance harness for the MAXLEN/SEQS knob-wiring patch (merlin-engine).

Uses the repo's own launch.build() resolver — pure, no GPU, no execution — to
prove two things across every profile of every registry:

  1. DEFAULTS: with no knobs in the environment, the resolved argv is
     byte-identical before and after the patch (nothing changes for anyone
     who does not export the knobs).
  2. KNOBS: with SEQS/MAXLEN exported, the --max-num-seqs / --max-model-len
     flags carry the exported values on the Flash-Next and 27B paths (they
     already did on SVL and Ornith).

Usage:
  python3 tools/merlin/knob_invariance.py save <file.json>    # capture argvs
  python3 tools/merlin/knob_invariance.py diff <a.json> <b.json>  # compare
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import launch as L  # noqa: E402
import registry as R  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
CKPT = "/nonexistent-checkpoint"  # build() does not read the checkpoint dir

MODELS = ["flash-next", "27b", "svl", "ornith"]

BASE_ENV = {
    "TREE": str(REPO),
    "VENV": str(REPO / "venv"),
    "CACHE_ROOT": str(REPO / ".vllm-cache"),
}


def profile_ids(model: str) -> list[str]:
    return [p["id"] for p in R.load_registry(model)["profiles"]]


def resolve_all(env: dict[str, str]) -> dict[str, object]:
    out = {}
    for model in MODELS:
        for pid in profile_ids(model):
            key = f"{model}/{pid}"
            try:
                plan = L.build(model, pid, CKPT, dict(env), [], repo=REPO)
                out[key] = plan.argv()
            except Exception as e:  # noqa: BLE002 -- record, don't abort the sweep
                out[key] = f"ERROR: {type(e).__name__}: {e}"
    return out


def main() -> int:
    if len(sys.argv) >= 3 and sys.argv[1] == "save":
        data = {"defaults": resolve_all(BASE_ENV)}
        if len(sys.argv) >= 4 and sys.argv[3] == "knobs":
            env = dict(BASE_ENV)
            env.update({"SEQS": "8", "MAXLEN": "131072"})
            data["knobs"] = resolve_all(env)
        Path(sys.argv[2]).write_text(json.dumps(data, indent=1, sort_keys=True))
        n = len(data["defaults"])
        print(f"saved {n} profile resolutions to {sys.argv[2]}"
              + (" (plus knob-variant)" if "knobs" in data else ""))
        return 0
    if len(sys.argv) >= 4 and sys.argv[1] == "diff":
        a = json.loads(Path(sys.argv[2]).read_text())
        b = json.loads(Path(sys.argv[3]).read_text())
        da, db = a["defaults"], b["defaults"]
        diffs = [k for k in da if da[k] != db.get(k)]
        if diffs:
            print(f"DEFAULTS DIFFER on {len(diffs)} profiles:")
            for k in diffs[:5]:
                for x, y in zip(da[k], db[k]):
                    if x != y:
                        print(f"  {k}: {x!r} -> {y!r}")
                        break
            return 1
        print(f"defaults identical on all {len(da)} profiles")
        for name in ("knobs",):
            if name in a and name in b:
                xb = b[name]
                changed = [k for k in xb if xb[k] != da.get(k)]
                print(f"{name} variant: {len(changed)} profiles changed argv")
                for k in sorted(changed)[:6]:
                    argv = xb[k]
                    seqs = argv[argv.index("--max-num-seqs") + 1]
                    mx = argv[argv.index("--max-model-len") + 1]
                    print(f"  {k}: --max-model-len={mx} --max-num-seqs={seqs}")
        return 0
    print(__doc__)
    return 2


if __name__ == "__main__":
    sys.exit(main())
