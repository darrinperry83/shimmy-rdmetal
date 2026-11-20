#!/usr/bin/env python3
"""Build and run the AArch64 dma-buf import round-trip guest regression.

The probe runs inside a disposable shim root (never a live rootfs): it creates
a dumb BO, exports it to a dma-buf fd (PRIME_HANDLE_TO_FD), re-imports that fd
(PRIME_FD_TO_HANDLE), and verifies the two mmap()s alias the same host backing.

Locate shim_cli at the repo root; when this test lives in a driver-only
subtree (rdmetal/tests), fall back to the enclosing checkout root.
"""
import os
import resource
import subprocess
import tempfile
from pathlib import Path

repo = Path(__file__).resolve().parent.parent

def find_shim(start: Path) -> Path:
    for d in [start] + list(start.parents):
        candidate = d / "shim_cli"
        if candidate.is_file():
            return candidate
    raise SystemExit(f"shim_cli not found under {start}")

shim = find_shim(repo)

(repo / "build").mkdir(exist_ok=True)
with tempfile.TemporaryDirectory(prefix="dmabuf-import-", dir=repo / "build") as tmp:
    root = Path(tmp)
    (root / "bin").mkdir()
    probe = root / "bin" / "probe"
    subprocess.run([
        "clang", "--target=aarch64-linux-gnu", "-nostdlib", "-static-pie",
        "-fuse-ld=lld", "-Wl,-e,_start",
        str(repo / "tests" / "guest-dmabuf-import.c"), "-o", str(probe),
    ], check=True)

    def no_core():
        resource.setrlimit(resource.RLIMIT_CORE, (0, 0))

    env = os.environ.copy()
    env["RDDRM"] = "1"  # render node is on by default; keep the arm explicit

    result = subprocess.run([str(shim), str(root), "/bin/probe"],
                            capture_output=True, text=True, timeout=30,
                            preexec_fn=no_core, env=env)
    if result.returncode or "DMABUF_IMPORT_OK" not in result.stdout:
        raise SystemExit(
            f"dma-buf import guest failed ({result.returncode}):\n"
            f"{result.stdout}{result.stderr[-2000:]}")
    print(result.stdout.strip())
