// Standalone, long-running StorageServer binary (Phase 5: data-only half of
// the split, see docs/PHASE_5.md). Binds, announces itself to the
// ControllerServer via HANDSHAKE (shard_index + its own bound address), then
// prints its bound address to stdout on one line and blocks until
// SIGTERM/SIGINT.
//
// Usage: storage_server <bind_address> <shard_index> <controller_address> [server_id]
//   e.g. storage_server tcp://127.0.0.1:0 0 tcp://127.0.0.1:5555 shard-0

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "transferqueue/RpcClient.h"
#include "transferqueue/StorageManager.h"
#include "transferqueue/StorageServer.h"

namespace {
std::atomic<bool> g_stop{false};
void handle_signal(int) { g_stop = true; }
} // namespace

int main(int argc, char** argv) {
    if (argc < 4) {
        std::cerr << "usage: " << argv[0] << " <bind_address> <shard_index> <controller_address> [server_id]\n";
        return 1;
    }
    std::string address = argv[1];
    std::int32_t shard_index = std::stoi(argv[2]);
    std::string controller_address = argv[3];
    std::string server_id = argc > 4 ? argv[4] : ("shard-" + std::to_string(shard_index));

    std::signal(SIGTERM, handle_signal);
    std::signal(SIGINT, handle_signal);

    auto storage = std::make_shared<tq::SimpleStorageManager>();
    tq::StorageServer server(server_id, shard_index, storage);

    std::string bound = server.start(address);

    tq::TransferQueueRpcClient controller_client(server_id, controller_address);
    controller_client.announce_shard(shard_index, bound);

    std::cout << bound << std::endl; // launcher reads this line to learn the real address

    while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    server.stop();
    return 0;
}
