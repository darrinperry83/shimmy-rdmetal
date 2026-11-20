#!/usr/bin/env python3
"""Regression: GBM swapchain dumb-buffer allocation and mmap in a disposable guest.

Builds tests/guest-gbm-bo.c as a bare-metal static-PIE aarch64 binary (raw svc #0
syscalls, no libc) and runs it under shim_cli in a private socket group. The guest
must print GBM_BO_OK, which exercises MODE_CREATE_DUMB, MODE_MAP_DUMB (non-zero
offset) and mmap write/read-back on the shim's synthetic DRM render node.
"""
from pathlib import Path
import os
import resource
import subprocess
import tempfile


def no_core():
    resource.setrlimit(resource.RLIMIT_CORE, (0, 0))


repo = Path(__file__).resolve().parent.parent
with tempfile.TemporaryDirectory(prefix="gbm-bo-group-") as group:
    with tempfile.TemporaryDirectory(prefix="gbm-bo-root-") as root_tmp:
        root = Path(root_tmp)
        (root / "bin").mkdir()
        guest = root / "bin/gbm-bo"
        subprocess.run([
            "clang", "--target=aarch64-linux-gnu", "-nostdlib", "-static-pie",
            "-fuse-ld=lld", "-Wl,-e,_start",
            str(repo / "tests/guest-gbm-bo.c"), "-o", str(guest),
        ], check=True, timeout=30)

        env = os.environ.copy()
        env.update(RDDRM="1", RDSOCKETGROUP=group)
        result = subprocess.run([str(repo / "shim_cli"), str(root), "/bin/gbm-bo"],
                                capture_output=True, text=True, timeout=30,
                                preexec_fn=no_core, env=env)
        if result.returncode or "GBM_BO_OK" not in result.stdout:
            raise RuntimeError(
                f"GBM BO probe failed rc={result.returncode}:\n"
                f"stdout={result.stdout}\nstderr={result.stderr[-1200:]}")

print("GBM_BO_OK")
