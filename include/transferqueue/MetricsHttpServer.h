#pragma once

// Minimal HTTP endpoint for Prometheus scraping: every GET (any path) is
// answered with render()'s text in the Prometheus exposition format. Plain
// POSIX sockets, one connection at a time -- a scrape every few seconds is
// all it serves.

#include <atomic>
#include <functional>
#include <string>
#include <thread>

namespace tq {

class MetricsHttpServer {
public:
    explicit MetricsHttpServer(std::function<std::string()> render);
    ~MetricsHttpServer();

    MetricsHttpServer(const MetricsHttpServer&) = delete;
    MetricsHttpServer& operator=(const MetricsHttpServer&) = delete;

    // Binds 0.0.0.0:port (0 = ephemeral) and serves in a background thread.
    // Returns the bound port.
    int start(int port);
    void stop();

private:
    std::function<std::string()> render_;
    int listen_fd_ = -1;
    std::atomic<bool> running_{false};
    std::thread thread_;
};

}
