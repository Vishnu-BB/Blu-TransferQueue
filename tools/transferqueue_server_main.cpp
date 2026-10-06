// Standalone, long-running TransferQueueServer binary -- for launching via
// an external process manager (Ray, a shell script, systemd, ...) rather
// than from a test harness, which always runs a fixed sequence and exits.
// Prints its bound address to stdout on one line (so a launcher doesn't
// need to pre-know the port -- supports "tcp://127.0.0.1:0" for an
// ephemeral one), then blocks until SIGTERM/SIGINT.
//
// Usage: transferqueue_server <bind_address> [server_id]
//   e.g. transferqueue_server tcp://127.0.0.1:0 shard-0

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "transferqueue/Controller.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/Server.h"
#include "transferqueue/StorageManager.h"

namespace {
std::atomic<bool> g_stop{false};
void handle_signal(int) { g_stop = true; }
} // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::cerr << "usage: " << argv[0] << " <bind_address> [server_id]\n";
        return 1;
    }
    std::string address = argv[1];
    std::string server_id = argc > 2 ? argv[2] : "server";

    std::signal(SIGTERM, handle_signal);
    std::signal(SIGINT, handle_signal);

    auto controller = std::make_shared<tq::TransferQueueController>();
    auto storage = std::make_shared<tq::SimpleStorageManager>();
    auto sampler = std::make_shared<tq::FifoSampler>();
    tq::TransferQueueServer server(server_id, controller, storage, sampler);

    std::string bound = server.start(address);
    std::cout << bound << std::endl; // launcher reads this line to learn the real address

    while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    server.stop();
    return 0;
}
