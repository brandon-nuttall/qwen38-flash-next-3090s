# Flash-Next v2.5.2 (Merlin engine v2.5.2) — package manifest (generated 2026-10-12 from the release tree)

Release assets (hashes from `release/v2.5/SHA256SUMS.v2.5.2`):

| asset | sha256 |
|---|---|
| `v2.5.2-from-upstream-v0.30.0.bundle` (uncompressed; the asset is the `.gz`; attached to the `v2.5.2` release) | `ab70e1a39768d4ae016f7f9f0ae99165c20c0e0136579f557084218cef8bd685` |
| `v2.5.2-from-upstream-v0.30.0.bundle.gz` (as downloaded) | `b154901627a985ce5469d4aebc2a15dbd07bb2c3e5c38f9517bdde9ae55ee0e2` |
| `build-artifacts-sm86-py313-cu130-v2.2.0.tar.gz` (v2.2.0's asset, reused unchanged; attached to the `v2.2.0` release) | `cdca5ffcb3003d15e00896f0969134e16d7bfbe0a3c6dfe10698bec32573f592` |
| `build-artifacts.list` | `2befac68d349aea26e0b35f59f4d04eda6c652ae7037f48fab8920c33ab5e432` |

The bundle's prerequisite is the public vLLM tag `v0.30.0` (`ced6857afa0ea7b2e3f0846a62e1394e90f15607`); it carries
`refs/heads/v2.5.2-public` = `de6e2c2fe9e02a9f70a5ff6f313aa45abc43b84c`, tree
`05e6d1a583e102b32a133ce34519865cacf32c45`, 98 commits (v2.2.0's 58, unchanged, plus 39 for v2.5.1 and 1 for v2.5.2:
the guard reader idle quiesce, `vllm/envs.py`, `vllm/v1/worker/gpu_worker.py` and `vllm/v1/worker/guard_telemetry.py`).
The 40 commits after v2.2.0 change no compiled code: Python, plus comment-only edits in three C/CUDA files (no CMake
or build file), so the compiled ops are v2.2.0's: the tarball
holds the 23 paths of `build-artifacts.list` (2,005 files), the two own extensions (`_C_stable_libtorch`
`6cb8bc7770112a9b…`, `_moe_C_stable_libtorch` `66acb123e71ba0c7…`; glibc 2.34 or newer) and the stock vLLM 0.30.0
wheel's other build products.

Combined diff: `release/v2.5/v2.5.2-combined.diff` = the diff from `v0.30.0` to the release commit (212 files);
`git apply --index` on `v0.30.0` yields the tree `05e6d1a5…`. Reference patch series: 98 files under
`release/v2.5/patches/` (0001..0097 are v2.5.1's content, re-exported as n/98; 0098 is new), hashed in `SHA256SUMS.v2.5.2`; `git am --keep-non-patch` on `v0.30.0`
(with `core.autocrlf=false`) replays them to the same tree. Environment pins: 196, identical to v2.2.0's except
`transformers==5.18.0` (was 5.17.0).

Files in this tree for v2.5.2 (sha256 of the committed, LF-normalised content — `git show <commit>:<path> | sha256sum`;
a CRLF checkout on Windows will hash differently). `release/v2.5/SHA256SUMS.v2.5.2` is hashed below it; this manifest
cannot hash itself.

| file | sha256 |
|---|---|
| `README.md` | `d492af1da0d737f7299cba8373c64b30f9e0c65d4ad788553b9a605b79f2318d` |
| `NOTICE` | `b5c2a57a51f793671d0a72107bc0759647873d3e19a8fa0a8ffed265feba9156` |
| `LICENSE` | `cfc7749b96f63bd31c3c42b5c471bf756814053e847c10f3eb003417bc523d30` |
| `LICENSE.qwen-community-1.0` | `465dcafcac7d3542a0cfc150fc550e0b7228c22ee68c8f1f81db942e7c658cb4` |
| `upstream/PIN-v2.5` | `6babc4461fffbf9b01628a9587a33cfc3b27eb71b5fc636de0ff2c933aa179f3` |
| `scripts/build-v2.5.sh` | `ac8e67e4862fc5032791269c8d03c6a549b68d2fe0f459615d37030ca1f43148` |
| `scripts/serve-v2.5.sh` | `3059569d1cc0fa4b2385b9a65eee9ecbe1d64bd801f3cd3da42203bd455a8e0f` |
| `scripts/topo_select.py` | `5ec2af2f8fdf179e2b7ab2ec8d29a0633e5570fa780c5c51c5bb11b159329ffc` |
| `docs/topology-selector.md` | `b67592ffa0dad04774413ee610acb3a51d7c71741235690ffd7d32dd428965f5` |
| `docs/history.md` | `4c2b83a3932a5693432f051ae6700bbf7dcb5133814a6de1fb3d4e037919c333` |
| `CHANGELOG.md` | `5a99099b56d6c334c32bd26fafcb550d4578dafc03966663ee7a71fcb593ef0c` |
| `docs/reference.md` | `a87d14e566cdfdca46d8a334436040709e741a46bb719c279a5cadfa0b884dc3` |
| `docs/images/flashnext-v2.2-layout.png` | `f76b34f6710c9ab58bce62dbcede8c1cb63e31246060e989369405cd8a6503c3` |
| `docs/images/flashnext-nvlink-pairing.png` | `f0e5c901d853ec2396db29082d6b2d6edaeeac20dd6463e351c7bfb4e9cd13fc` |
| `Dockerfile` | `d79a423c2e1263526ebe369059003c7733f1a8e0ab0d664824933f109d399551` |
| `docker-compose.yml` | `aebc361814fc27a5a754dee98042d546ac4d16041f330ed2b7aba5816cff552f` |
| `scripts/docker-entrypoint.sh` | `deaee5fc288fb548fbad2c39cc10ec1f6009dc6785ae9b1f96650ef86b04b17c` |
| `lxc/pve-create.sh` | `41cd487badaebf99a5de2d925f99f00399029f2837b654b6ba3c433b294a882d` |
| `lxc/provision.sh` | `8b50dda9102c1934211856458c767838b2633df414525dd1c7fb941e70a21f36` |
| `lxc/nvidia-prestart.sh` | `9576cce20a6bdb1ba517f4b2473077858e346a1223d8b5f15ea7e03cf19de690` |
| `lxc/flash-next.service` | `a306b9e3760945dad47096be81173352f5dcb8ba7c9aea5615435b85397003e9` |
| `.dockerignore` | `67779c162e163c24be2f1e41c9c29c980ca1fc013090f5cdf063921d3cd1df71` |
| `scripts/make-e1-config.py` | `33f6ff5789e4d9ff6649706e22964e551cfed809c00a9d578fcc83a83b9a5646` |
| `scales/qsa_kv_scales_262k.json` | `5cbe6ae87ed9d2dc97453a9a7b4967d0e0d32a0e6992ec260dc2c54b96305f67` |
| `release/v2.5/requirements-pinned.txt` | `3e108cf03a6ef6c8bae7f376f5e2764b8ef7e3c0db6ea8bba164589fdf8dc041` |
| `release/v2.5/build-artifacts.list` | `2befac68d349aea26e0b35f59f4d04eda6c652ae7037f48fab8920c33ab5e432` |
| `benchmarks/2026-09-24/BENCH-CARD.md` | `02971fb8710be0156428d75230f8158ad858bba3ca260c69cf4509e8f1faea00` |
| `benchmarks/2026-09-24/flashnext-v2.2.0-ctx-pp-tg-itl.png` | `a954b5af737868a398162b2890b9bd5f2715cce73939a496c381bc652921c723` |
| `benchmarks/2026-10-05/BENCH-CARD_OLD.md` | `ad6b617beef31c0e459d8a163cf23c493ff1e1914530007ecd13d5dcefe20e56` |
| `benchmarks/2026-10-05/flashnext-v2.5.0-ctx-pp-tg-itl_OLD.png` | `ee90d9efdfc99750a380c5f36c8b2daeaa4bbcfad5872c69d4f28bbb200451d1` |
| `benchmarks/2026-10-06/BENCH-CARD.md` | `cd187ea6fd4f46e5c8c0e9cd384a10806bf79a4a9bf5f2003145db89d941b7da` |
| `benchmarks/2026-10-06/flashnext-v2.5.1-ctx-pp-tg-itl.png` | `4832f93602fc2c16638901fa11aede507009d6299e0c7b920c2d679944b45061` |
| `benchmarks/2026-10-12/BENCH-CARD.md` | `2822472e6bb6c166c76046b8f6e27b0f37dc75674dbed3b59ea3304b4956f5cb` |
| `docs/images/flashnext-v2.5-layout.png` | `99b55769952494bd852b1c19248353a44d30893399a17dbcb0674114f4b5ccbe` |
| `docs/images/flashnext-v2.5-summary.png` | `ec9b0474c61f21263807cb55925fc5cb4c63593f844d9de0d80ff2dc764aff6d` |
| `release/v2.5/v2.5.2-combined.diff` | `bdddf25d5289a38695ace612b1e3d02caccea039073ea8473bbfb6b0c9def98b` |
| `release/v2.5/SHA256SUMS.v2.5.2` | `bc3b1caf2b56f56cdfb5858ed3ef392c5680ea09e573366ef68ba48dfb33040c` |

The v2.2.0, v2.0.1 and v1 files stay in the tree unchanged (their manifests are `release/v2.2/PACKAGE-MANIFEST.md` and
`release/v2/PACKAGE-MANIFEST.md`); the container and LXC recipes (`Dockerfile`, `docker-compose.yml`,
`scripts/docker-entrypoint.sh`, `lxc/`) build v2.5.2 (listed above).

Checkpoint: `Qwen3.8-Flash-Next-W4A16-Merlin` from Hugging Face plus the one config key added by
`scripts/make-e1-config.py`, unchanged from v2.0.1; shard hashes are the checkpoint's own (not republished here).
