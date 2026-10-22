#!/usr/bin/env bash
# Deprecated entry point (the v2.5 build name): forwards to scripts/build.sh (the wheel builder); the serving
# environment is created by release/install-env.sh. Removed in the release after Merlin 3.0.0.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
echo "note: scripts/build-v2.5.sh is deprecated; forwarding to scripts/build.sh (serving venv: release/install-env.sh)" >&2
exec bash "$here/build.sh" "$@"
