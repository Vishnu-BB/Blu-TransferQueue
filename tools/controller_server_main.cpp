// Standalone, long-running ControllerServer binary (Phase 5: metadata-only
// half of the split, see docs/PHASE_5.md). Prints its bound address to
// stdout on one line, then blocks until SIGTERM/SIGINT.
//
// Usage: controller_server <bind_address> [server_id]
//   e.g. controller_server tcp://127.0.0.1:0 controller

#include <atomic>
#include <chrono>
#include <csignal>
#include <iostream>
#include <memory>
#include <string>
#include <thread>

#include "transferqueue/Controller.h"
#include "transferqueue/ControllerServer.h"
#include "transferqueue/Sampler.h"

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
    std::string server_id = argc > 2 ? argv[2] : "controller";

    std::signal(SIGTERM, handle_signal);
    std::signal(SIGINT, handle_signal);

    auto controller = std::make_shared<tq::TransferQueueController>();
    auto sampler = std::make_shared<tq::FifoSampler>();
    tq::ControllerServer server(server_id, controller, sampler);

    std::string bound = server.start(address);
    std::cout << bound << std::endl; // launcher reads this line to learn the real address

    while (!g_stop.load()) {
        std::this_thread::sleep_for(std::chrono::milliseconds(200));
    }

    server.stop();
    return 0;
}
