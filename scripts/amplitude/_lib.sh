# Common script harness — sourced by every NN_*.sh script.
# Sets up tee'd logging into scripts/amplitude/logs/<script>.<UTC-stamp>.log,
# defines say(), expect(), pass()/fail() helpers, and prints a tail-of-log on
# error so failures are debuggable from the terminal alone.

set -euo pipefail

# Make user-installed binaries (uv, etc.) available across all amplitude
# scripts even when the interactive shell hasn't been re-sourced. The export
# in 20_setup.sh only persists for that subshell — putting it here covers
# every NN_*.sh that comes after.
if [ -d "$HOME/.local/bin" ] && [[ ":$PATH:" != *":$HOME/.local/bin:"* ]]; then
    export PATH="$HOME/.local/bin:$PATH"
fi

_SCRIPT_NAME="$(basename "${BASH_SOURCE[1]}" .sh)"
_SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[1]}")" && pwd)"
LOG_DIR="${LOG_DIR:-${_SCRIPT_DIR}/logs}"
mkdir -p "${LOG_DIR}"
LOG="${LOG_DIR}/${_SCRIPT_NAME}.$(date -u +%Y%m%dT%H%M%SZ).log"

# All stdout/stderr from the calling script gets tee'd into $LOG.
exec > >(tee -a "${LOG}") 2>&1

# Print tail of log on error so the user can paste failures back without
# having to find the file.
trap '_rc=$?; echo; echo "FAIL (exit $_rc) — last 30 lines:"; tail -30 "${LOG}" | sed "s/^/    /"; echo "(full log: ${LOG})"; exit $_rc' ERR

say()    { echo "==> $*"; }
expect() { echo "    EXPECT: $*"; }
pass()   { echo; echo "PASS — ${_SCRIPT_NAME}"; echo "log:  ${LOG}"; }

echo "=== ${_SCRIPT_NAME} @ $(date -u +%FT%TZ) ==="
echo "host=$(hostname -s) user=${USER:-?} cwd=$(pwd)"
echo "log=${LOG}"
echo
