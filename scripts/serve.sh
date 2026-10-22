#!/usr/bin/env bash
# One-release shim (Merlin 3.0.0): the Flash-Next launcher moved to models/flash-next/serve.sh. Same arguments,
# same environment. Removed in the release after Merlin 3.0.0.
here="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
echo "note: scripts/serve.sh moved to models/flash-next/serve.sh; forwarding" >&2
exec bash "$here/models/flash-next/serve.sh" "$@"
