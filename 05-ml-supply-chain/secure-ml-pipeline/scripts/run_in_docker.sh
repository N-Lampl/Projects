#!/usr/bin/env bash
# Detonate the BENIGN pickle PoC inside a locked-down container so the marker
# write happens in an isolated, network-less, read-only sandbox.
#
#   bash scripts/run_in_docker.sh [path/to/malicious_model.pkl]
#
# Hardening flags:
#   --network none   no network access at all
#   --read-only      root filesystem is read-only
#   --tmpfs /tmp     a small writable tmpfs ONLY for the marker file
#   --rm             container removed after exit
#   --cap-drop ALL   drop all Linux capabilities
#
# If docker is not installed, we SKIP with a warning and tell you how to use the
# pure-python opcode scanner instead (which never executes the payload).
set -euo pipefail

PROJECT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
POC_PATH="${1:-$PROJECT_DIR/data/poc/malicious_model.pkl}"
IMAGE="secure-ml-pipeline-sandbox"

if ! command -v docker >/dev/null 2>&1; then
  echo "WARNING: docker not installed - SKIPPING sandboxed detonation." >&2
  echo "         Statically inspect the PoC WITHOUT executing it instead:" >&2
  echo "         python scripts/build_poc.py   # runs the opcode scanner" >&2
  exit 0
fi

if [[ ! -f "$POC_PATH" ]]; then
  echo "PoC not found at $POC_PATH - building it first..." >&2
  python "$PROJECT_DIR/scripts/build_poc.py" --out "$POC_PATH"
fi

echo "Building sandbox image ($IMAGE)..."
docker build -t "$IMAGE" -f "$PROJECT_DIR/docker/Dockerfile" "$PROJECT_DIR"

echo "Detonating PoC inside locked-down container..."
echo "  flags: --network none --read-only --tmpfs /tmp --cap-drop ALL"
docker run --rm \
  --network none \
  --read-only \
  --tmpfs /tmp:rw,size=8m \
  --cap-drop ALL \
  --security-opt no-new-privileges \
  -v "$POC_PATH:/poc/malicious_model.pkl:ro" \
  "$IMAGE" \
  python /sandbox/detonate.py /poc/malicious_model.pkl /tmp/PWNED_DEMO

echo "Done. The marker write happened ONLY inside the now-destroyed container."
echo "Nothing touched your host filesystem outside the ephemeral tmpfs."
