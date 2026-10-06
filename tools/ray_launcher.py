"""Launches Blu-TransferQueue's standalone server binary (tools/
transferqueue_server_main.cpp, built via `make server_bin`) as a Ray actor.

Ray only ever manages the OS process -- start it, track whether it's alive,
stop it, (and in a real cluster) place it on a chosen node. Our C++ code is
completely unaware of Ray; this is the same relationship mpirun/BluRun
already have with the same binary, just with Ray as the launcher instead.

Why not Ray's C++ actor SDK directly: tried it first. Ray's C++ SDK requires
building the actor code into a plugin .so that a spawned Ray *worker*
process dlopen()s at runtime. Built against this machine's plain g++, that
dlopen() segfaults inside Ray's own FunctionHelper::LoadDll() -- a toolchain
ABI mismatch between our compiler and the exact pinned Bazel toolchain
Ray's prebuilt libraries expect. Fixing that properly means adopting Bazel
(not installed here, no simple one-command install path) as a second build
system alongside our Makefile, for one feature. This subprocess-based
approach avoids that fragility entirely -- see docs/RAY_LAUNCHER.md.
"""

import os
import subprocess
import time

# transferqueue_server's rpath (../Tensor-Implementations/lib, baked in by
# the Makefile) is relative to the *current working directory*, not the
# binary's own location ($ORIGIN) -- it only resolves when run from
# Blu-TransferQueue/, same cwd the Makefile always runs it from. Launched by
# something with a different cwd (a Ray worker, in particular), libtensor.so
# fails to resolve. Set LD_LIBRARY_PATH explicitly here rather than change
# the Makefile's rpath convention, which works fine for every other
# existing use (make smoke/test_tensor/test_rpc/test_distributed all run
# from Blu-TransferQueue/ already).
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_BLUTRAIN_DIR = os.path.abspath(os.path.join(_THIS_DIR, "..", ".."))
_EXTRA_LIB_DIRS = [
    os.path.join(_BLUTRAIN_DIR, "Tensor-Implementations", "lib"),
    os.path.join(_BLUTRAIN_DIR, "Profiler", "lib"),
]


class TransferQueueServerActor:
    """Wraps one `transferqueue_server` OS process. Decorate with
    `ray.remote` at the call site (not here) so this stays importable and
    testable without Ray initialized."""

    def __init__(self, binary_path: str):
        self._binary_path = binary_path
        self._process: subprocess.Popen | None = None
        self._address: str | None = None

    def start(self, bind_address: str, server_id: str = "server") -> str:
        env = dict(os.environ)
        existing = env.get("LD_LIBRARY_PATH", "")
        env["LD_LIBRARY_PATH"] = ":".join(_EXTRA_LIB_DIRS + ([existing] if existing else []))

        self._process = subprocess.Popen(
            [self._binary_path, bind_address, server_id],
            stdout=subprocess.PIPE,
            text=True,
            env=env,
        )
        # The binary prints its bound address as the first stdout line --
        # resolves "tcp://127.0.0.1:0" to the real ephemeral port, so the
        # caller doesn't have to pre-know or parse it out another way.
        line = self._process.stdout.readline().strip()
        if not line:
            raise RuntimeError(
                f"transferqueue_server exited before printing its address "
                f"(exit code {self._process.poll()})"
            )
        self._address = line
        return self._address

    def is_alive(self) -> bool:
        return self._process is not None and self._process.poll() is None

    def stop(self, timeout: float = 5.0) -> None:
        if self._process is None:
            return
        self._process.terminate()
        try:
            self._process.wait(timeout=timeout)
        except subprocess.TimeoutExpired:
            self._process.kill()
            self._process.wait()
        self._process = None
