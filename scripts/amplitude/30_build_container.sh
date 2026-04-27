#!/bin/bash
# WHERE: Amplitude login-node, inside the repo.
# WHAT:  Build the Apptainer image hpc/run_container.sif from run_container.def.
# WHY:   The sbatch script bind-mounts and runs this container — it must exist
#        on the login node before submission.
# COST:  ~3–8 min depending on Amplitude's apt mirror speed.
#
# Apptainer build sometimes needs --fakeroot on shared HPCs (no root daemon).
# This script tries the plain build first, then --fakeroot, then prints a
# clear message if both fail (in which case ask HPC support how local users
# build images).

source "$(dirname "$0")/_lib.sh"

REPO_ROOT="$(cd "$(dirname "$0")/../.." && pwd)"
cd "${REPO_ROOT}"

if [ -f hpc/run_container.sif ]; then
    say "Image already exists — sha & mtime"
    ls -la hpc/run_container.sif
    sha256sum hpc/run_container.sif | head -1
    echo "    (delete hpc/run_container.sif and re-run if you changed run_container.def)"
    pass
    exit 0
fi

if ! command -v apptainer >/dev/null 2>&1; then
    echo "ERROR: apptainer not on PATH. Try: module load apptainer ; then re-run."
    exit 1
fi

say "Trying plain apptainer build"
if apptainer build hpc/run_container.sif hpc/run_container.def 2>&1 | tail -20; then
    pass
    exit 0
fi

say "Plain build failed — trying --fakeroot"
if apptainer build --fakeroot hpc/run_container.sif hpc/run_container.def 2>&1 | tail -20; then
    pass
    exit 0
fi

cat <<'EOF'

ERROR: Both plain and --fakeroot apptainer builds failed.

This usually means Amplitude restricts container builds to admin nodes.
Options to unstick:
  1. Build the image somewhere else (your laptop with apptainer installed,
     or a separate build node) and rsync it to ~/projects/.../hpc/run_container.sif
  2. Ask HPC support: "what's the recommended way for users to build
     Apptainer images?" — some clusters provide /opt/apptainer-build/ or
     a `apptainer-build-as-root` helper.
EOF
exit 1
