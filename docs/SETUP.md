# setup.sh: install and serve in one command

```bash
git clone https://github.com/halt95/merlin-engine.git && cd merlin-engine && ./setup.sh
```

`setup.sh` checks the host, asks a few questions (Enter takes the recommended answer, computed from the GPUs and RAM it
finds), shows every download with its size, installs the engine from the prebuilt release wheel (about 1.5 minutes;
nothing system-wide, no sudo), downloads and verifies the checkpoint, and starts the server in the foreground.
Ctrl+C stops it; `./setup.sh` again starts it straight away with the saved answers and asks nothing;
`git pull && ./setup.sh` updates the engine without re-downloading the model.

New to serving? [How to serve](how-to-serve.md) walks through it end to end, from the first run to a networked endpoint.

## What a run looks like

```
Merlin 3.0.0 (by Halt95) setup (~/merlin-engine)
Host: GPUs 4 x NVIDIA GeForce RTX 3090 (24 GB); P2P on; driver 595.84; RAM 96 GiB
Model (Enter = recommended):
  1) flash-next         Qwen3.8 Flash-Next W4A16 (the reference recipe)                         115.3 GiB
  2) swift1.5           Swift 1.5 Flash-Next W4A16 (UkisAI fine-tune)                           115.3 GiB
  3) swift1.5-huihui    Swift 1.5 Huihui Flash-Next W4A16 (abliterated)                         115.3 GiB
  4) 27b                Qwen3.8 27B W4A16 (dense; 1, 2 or 4 cards)                               15.9 GiB
  5) 27b-thinkingcap    ThinkingCap Qwen3.8 27B W4A16 (BottleCap AI; separate weight licence)    15.9 GiB
  6) nemotron-super-vl  Nemotron 3.5 Super VL 120B-A12B W4A16 (4 cards)
  7) ornith1.5          Ornith 1.5 122B-A10B W4A16 (4 x 24 GB; with or without P2P)              66.4 GiB
  recommended: flash-next (4 x GeForce RTX 3090, P2P on, 96 GiB RAM: fn-quad-3090 [tested] fits with the qualified RAM)
Model [flash-next]:
Layout (Enter = recommended):
  1) default                fn-quad-3090 [tested] on GPUs 0,1,2,3  <- recommended
  recommended: default (what merlin serve picks)
Layout [default]:
Models folder (an existing download is reused) [~/models/flash-next]:
Access:
  1) local    this machine only (127.0.0.1)
  2) network  every host that can reach this machine (0.0.0.0), an API key is required
Access [local]:
Port [8000]:
Your answers (nothing is downloaded or saved yet):
  1) model   flash-next
  2) layout  default layout
  3) folder  ~/models/flash-next
  4) access  local (127.0.0.1)
  5) port    8000
Change an answer? (Enter = no, or 1-5) [no]:
Downloads:
  engine wheel                                                     217 MB
      https://github.com/halt95/merlin-engine/releases/download/merlin-v3.0.0/vllm-0.32.0+merlin.7002f56537-cp313-cp313-linux_x86_64.whl
  Python packages (222 from release/requirements-0.32.lock)       3.4 GiB
      PyPI via uv (cached after the first install)  (about 10.2 GiB installed)
  checkpoint flash-next                                         115.3 GiB
      hf download halt95/Qwen3.8-Flash-Next-W4A16-Merlin --revision cec53e9f... -> ~/models/flash-next  (40 files, 115.3 GiB in all, size from Hugging Face API)
  total                                                         118.9 GiB
Serve: HOST=127.0.0.1 PORT=8000 ~/merlin-engine/venv/bin/python ~/merlin-engine/merlin_cli.py serve flash-next ~/models/flash-next
Proceed with the downloads above? (y/n) [y]:
```

Then the engine installs (`release/install-env.sh` with the wheel, then `--verify`), `hf download` fetches the
checkpoint (resumable), every file is checked against `models/<model>/checkpoint.sha256`, `merlin doctor` prints the
hardware report, and the server starts:

```
Endpoint: http://127.0.0.1:8000/v1  (OpenAI-compatible; /v1/messages for /v1/messages clients)
Test (once the log says `Application startup complete`):
  curl -s http://127.0.0.1:8000/v1/chat/completions -H 'Content-Type: application/json' -d '{"model": "flash-next-v2", "messages": [{"role": "user", "content": "Say hello."}], "max_tokens": 32}'
Ctrl+C stops the server; ./setup.sh starts it again without questions.
```

## Host checks (before anything is downloaded)

Every failure prints the exact fix and stops. Nothing is installed system-wide; the fixes are the only lines that
mention `sudo`. The bootstrap itself (uv and Python 3.13 into `.merlin/`, about 60 MB, both checksum-verified) runs
before the Python checks because the checks are written in Python; a `--dry-run` fetches neither and runs on any
`python3` 3.10+ it finds, or says what a real run would fetch.

| Check | Requirement | Fix printed |
|---|---|---|
| platform | Linux x86-64 | (the wheel and the lock are built for it) |
| curl, tar, git, sha256sum | present | `sudo apt install curl tar git coreutils` |
| NVIDIA driver | `nvidia-smi` present, R580 or later (CUDA 13) | `sudo apt install nvidia-driver-580-open` (or the .run installer), reboot |
| GPUs | at least one card with a Merlin layout | `merlin doctor` explains what fits |
| compiler | GCC/G++ >= 13: `CC` / `CXX`, else the remembered pair, else `cc` / `c++`; a `gcc-13` / `g++-13` (or 14, 15) pair on PATH is found by itself and remembered in `.merlin/setup.json` | `sudo apt install gcc-13 g++-13`, then `./setup.sh` again |
| git | a git clone (not a ZIP), tracked tree clean | `git clone`; commit or revert changes |
| host RAM | Flash-Next: >= 69 GiB resident floor (96 GiB qualified); 27B / Super VL: a warning below 32 GiB | add RAM or `--model 27b` |
| disk | per filesystem: the wheel download, uv's package cache (up to 3.4 GiB), the new venv (about 10 GiB), the previous venv while the new one verifies, and the checkpoint's remaining bytes (a resume needs only what is still missing) | free space or `--models-dir` |
| models folder | empty, or this model's checkpoint; a folder holding another checkpoint (by its `config.json` or verified marker) is refused | `--models-dir` another folder |
| `--from-source` only | cargo, cmake, ninja | rustup; `sudo apt install cmake ninja-build` |

## Questions

| Question | Default (Enter) | Flag |
|---|---|---|
| model | recommended from the hardware: Flash-Next on 4 x 24 GB (or 2 x 48 GB) with >= 96 GiB RAM; otherwise the 27B family: ThinkingCap at TP4 on 4 x 3090-class 24 GB cards with P2P (the fastest 27B-family layout: 2026-10-20 bench cards, 1 stream, thinking on, 4K / 131K / 262K, ThinkingCap TP4 253 / 255 / 248 tok/s vs 27B TP4 250 / 235 / 215; separate weight licence, shown before the download), 27B elsewhere (2 cards, no P2P, other classes); the reason is shown | `--model flash-next\|swift1.5\|swift1.5-huihui\|27b\|27b-thinkingcap\|nemotron-super-vl\|ornith1.5` |
| layout | what `merlin serve` picks (the largest tested layout that fits), with every other tensor-parallel size and every `--instances N` partition the host allows listed; one exception: a single-card 27B pick on a multi-card class without P2P recommends `--tp 1 --instances N` over all idle cards, with the measured reason (4 x CMP 170HX: TP1 x 4 = 873 tok/s, TP2 x 2 = 523, TP4 = 371; each single-card server takes images up to 4 MP, the multi-card layouts 8 MP) -- this is setup's recommendation only, `merlin serve`'s own default is unchanged | `--tp N`, `--instances N` |
| models folder | `~/models/<model>`; an existing folder is reused and skipped when its checksums pass | `--models-dir DIR` |
| access | this machine only (127.0.0.1); network (0.0.0.0) generates an API key, shows it once, and keeps it in `.merlin/api-key` (mode 0600); clients send `Authorization: Bearer <key>` | `--host 127.0.0.1\|0.0.0.0`, `--api-key KEY` |
| port | the model's default: 8000 for Flash-Next / Super VL, 8100 for 27B | `--port P` |

After the last question setup lists the answers. A number asks that question again (another model also asks its own layout, folder and port again); Enter, or `n`, carries on. Answers given by flags are listed but not offered. Nothing is downloaded or saved before the download confirmation, so `n` there, or Ctrl+C at any point before it, leaves nothing behind.

Answers are saved per model: the folder, port and layout of Flash-Next are never applied to 27B. `--model 27b` after a
Flash-Next run takes 27B's own saved answers, or its defaults.

## Flags

| Flag | Effect |
|---|---|
| `--yes` / `-y` | no questions: the recommended (or given) answers, downloads confirmed |
| `--reconfigure` | ask again (the saved answers are the defaults) |
| `--dry-run` | print the checks, answers, download plan and serve command; download nothing, write nothing (not even the answers) |
| `--simulate SPEC` | run the checks and plan against preflight's fabricated hardware (`4x3090`, `2x3090:ram=64`, `4x4090:nop2p`, ...); implies `--dry-run` unless `--no-start` (nothing ever starts on simulated hardware) |
| `--no-start` | install and download, then stop before `merlin serve` |
| `--rehash` | hash every checkpoint file again even when the verified marker is current |
| `--from-source` | compile the engine (about 1.5 h on 8 cores) instead of taking the release wheel |
| `--help` | the option list |

Non-interactive example: `./setup.sh --yes --model 27b --models-dir /data/27b --host 0.0.0.0 --api-key "$KEY" --port 8100 --tp 2`.

Environment overrides: `MERLIN_WHEEL=<file>` installs a local copy of the release wheel, `MERLIN_WHEEL_URL=<url>` fetches
it from elsewhere (both still checked against `release/merlin-wheel.sha256`); `MERLIN_RELEASE_VERSION` overrides the
release version read from the `## Merlin <version>` heading of `docs/CHANGELOG.md`; `MERLIN_UV_VERSION` another uv (checked against its published `.sha256` only);
`HF_TOKEN` is passed to the Hugging Face API and to `hf download` when set (no Merlin checkpoint needs it); `CC` / `CXX` name the compiler
pair (remembered).

## Downloads

The plan is shown with sizes before anything starts and confirmed once, after the questions (`--yes` skips the
question; a saved re-run asks nothing). An unknown size is shown as unknown, never as zero.

- uv (26 MB) and Python 3.13 (33 MB) into `.merlin/`, only when the host has none: uv's archive is checked against the
  release's published `.sha256` and, for the pinned version, the digest recorded in `setup.sh`;
- the engine wheel from the GitHub release `merlin-v<version>` named by `release/merlin-wheel.sha256`, verified against
  that sha256 (a wrong file is deleted, not installed); `curl -C -` resumes a partial download; the asset name with
  `+`, its `%2B` form and the `.` form GitHub may rename it to are all tried;
- the Python packages of `release/requirements-0.32.lock` (222 pins with hashes), fetched by uv and cached, so a
  reinstall takes them from the cache;
- the checkpoint, by the venv's `hf download` (resumable), pinned to one revision: the one `checkpoint.sha256` or
  `release/download-sizes.json` records. A checkpoint whose Hugging Face repository is not public yet
  (`pending_publication`: Super VL and Ornith 1.5 until their repositories go up) is served from a local copy only
  (`--models-dir`); nothing is downloaded from a moving `main`.

Checkpoint sizes come from the Hugging Face API, or from `release/download-sizes.json` when it is unreachable.

## Verification

After the download every file in `models/<model>/checkpoint.sha256` is hashed. A pass writes
`<models folder>/.merlin-verified.json` (the model, the checksum list's hash, each file's size and mtime). Later runs
skip the full re-hash while those match, and re-hash whenever the list, the file set, a size or an mtime changed, or
with `--rehash`. The trade-off: a corruption that keeps a file's size and mtime passes until `--rehash` (a 115 GiB
re-hash takes minutes, which is why the marker exists). A failure names the files; delete them and run again
(`hf download` fetches them anew).

`merlin pull <model> [--models-dir DIR] [--rehash] [--dry-run]` runs only the download and verification: no GPU,
driver or compiler check, no serving choice. It uses the engine venv's `hf` when there is one, else a small
`.merlin/hf` environment with the lock's huggingface-hub, so it works on a host that only downloads.

## Re-runs and updates

- `./setup.sh` again: the saved answers (`.merlin/setup.json`) are used, nothing is asked (not even the download
  confirmation), the venv is reused when `.merlin/state.json` records the same wheel sha, lock hash and commit, the
  checkpoint is skipped when its marker is current, and the server starts.
- `git pull && ./setup.sh`: a moved commit, changed lock or changed wheel reinstalls the engine: the working venv is
  moved aside, the new one is installed and verified, then the old one is removed; if the install or the verify
  fails, the old venv comes back and the engine still serves. The packages come from uv's cache; the model is never
  re-downloaded.
- `./setup.sh --reconfigure`: the questions again, with the saved answers as defaults.

## Where things live

| Path | Content |
|---|---|
| `.merlin/bin/` | uv (fetched, or a link to the one on PATH) and a `python3` link to the repo-local Python, for `release/install-env.sh` |
| `.merlin/python/` | the uv-managed Python 3.13 (`UV_PYTHON_INSTALL_DIR`) |
| `.merlin/downloads/` | the release wheel |
| `.merlin/setup.json` | the answers (model, host, compiler; folder, port and layout per model), mode 0600 |
| `.merlin/api-key` | the API key for network access, mode 0600 (`.merlin/` itself is 0700) |
| `.merlin/state.json` | the install record (wheel sha, lock hash, commit, venv) |
| `.merlin/hf/` | the download-only `hf` environment (`merlin pull` without the engine) |
| `venv/` | the engine environment `release/install-env.sh` created |
| `.build-source`, `.build-products`, `.build-products.sha256` | install-env's provenance and product hashes, in the checkout |
| `.vllm-cache/` | compile / Triton caches the launcher writes |
| `~/.cache/uv/` | uv's package cache (about 3.4 GiB; `uv cache clean` empties it) |
| `<models folder>` | the checkpoint and its `.merlin-verified.json` marker |

`.merlin/`, `venv/`, `.vllm-cache/` and the `.build-*` files are git-ignored.

## Uninstall

Delete `.merlin/`, `venv/`, `.vllm-cache/` and the `.build-*` files inside the checkout, the models folder
(`~/models/<model>` by default), and `~/.cache/uv` if you do not use uv otherwise. Nothing else was written: no
packages, services, shell profiles or system paths.

## Manual route

The pieces `setup.sh` drives are documented on their own: [release/install-env.sh](../release/README.md) (the venv),
`hf download` plus `sha256sum -c models/<model>/checkpoint.sha256` (the checkpoint), `merlin doctor` and
`merlin serve` ([README](../README.md#quick-start)).
