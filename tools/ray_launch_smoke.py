"""Runnable check for Ray-as-launcher: Ray starts N transferqueue_server
processes, we confirm each is actually listening on the address it
reported, then Ray stops them and we confirm they're actually gone.

Does NOT re-test put/get wire-protocol correctness -- that's already
proven by the C++ test suite (`make test_rpc`, `make test_distributed`).
This only proves Ray can launch, track, and cleanly stop our existing
binary, which is the actual thing this adds.

Run (from Blu-TransferQueue/, after `make server_bin`):
    .venv-ray/bin/python tools/ray_launch_smoke.py
"""

import socket
import sys
import time

import ray

from ray_launcher import TransferQueueServerActor


def can_connect(address: str, timeout: float = 5.0) -> bool:
    host_port = address.split("://", 1)[1]
    host, port = host_port.rsplit(":", 1)
    deadline = time.time() + timeout
    while time.time() < deadline:
        try:
            with socket.create_connection((host, int(port)), timeout=0.5):
                return True
        except OSError:
            time.sleep(0.1)
    return False


def main(binary_path: str) -> None:
    ray.init()

    Actor = ray.remote(TransferQueueServerActor)
    actors = [Actor.remote(binary_path) for _ in range(2)]

    addresses = ray.get(
        [a.start.remote("tcp://127.0.0.1:0", f"shard-{i}") for i, a in enumerate(actors)]
    )
    print(f"Ray launched servers at: {addresses}")

    for addr in addresses:
        assert can_connect(addr), f"Ray-launched server at {addr} is not reachable"
    assert all(ray.get([a.is_alive.remote() for a in actors])), "a Ray-launched server died early"

    ray.get([a.stop.remote() for a in actors])
    time.sleep(0.5)
    assert not any(ray.get([a.is_alive.remote() for a in actors])), "a server survived stop()"
    for addr in addresses:
        assert not can_connect(addr, timeout=1.0), f"{addr} is still accepting connections after stop()"

    ray.shutdown()
    print("ray_launch_smoke: PASS")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else "build/transferqueue_server")
