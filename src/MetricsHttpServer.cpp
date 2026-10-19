#include "transferqueue/MetricsHttpServer.h"

#include <netinet/in.h>
#include <poll.h>
#include <sys/socket.h>
#include <unistd.h>

#include <stdexcept>

namespace tq {

MetricsHttpServer::MetricsHttpServer(std::function<std::string()> render) : render_(std::move(render)) {}

MetricsHttpServer::~MetricsHttpServer() { stop(); }

int MetricsHttpServer::start(int port) {
    listen_fd_ = ::socket(AF_INET, SOCK_STREAM, 0);
    if (listen_fd_ < 0) throw std::runtime_error("MetricsHttpServer: socket() failed");
    int one = 1;
    ::setsockopt(listen_fd_, SOL_SOCKET, SO_REUSEADDR, &one, sizeof(one));
    sockaddr_in addr{};
    addr.sin_family = AF_INET;
    addr.sin_addr.s_addr = htonl(INADDR_ANY);
    addr.sin_port = htons(static_cast<std::uint16_t>(port));
    socklen_t len = sizeof(addr);
    if (::bind(listen_fd_, reinterpret_cast<sockaddr*>(&addr), sizeof(addr)) != 0 || ::listen(listen_fd_, 8) != 0 ||
        ::getsockname(listen_fd_, reinterpret_cast<sockaddr*>(&addr), &len) != 0) {
        ::close(listen_fd_);
        listen_fd_ = -1;
        throw std::runtime_error("MetricsHttpServer: cannot listen on port " + std::to_string(port));
    }

    running_ = true;
    thread_ = std::thread([this] {
        while (running_) {
            pollfd pfd{listen_fd_, POLLIN, 0};
            if (::poll(&pfd, 1, 200) <= 0) continue;  // 200 ms: how fast stop() is noticed
            int fd = ::accept(listen_fd_, nullptr, nullptr);
            if (fd < 0) continue;
            char request[1024];
            (void)::recv(fd, request, sizeof(request), 0);  // only GET is expected; content ignored
            std::string body;
            std::string status = "200 OK";
            try {
                body = render_();
            } catch (const std::exception& e) {
                status = "500 Internal Server Error";
                body = std::string("# metrics render failed: ") + e.what() + "\n";
            }
            std::string response = "HTTP/1.0 " + status +
                                   "\r\nContent-Type: text/plain; version=0.0.4\r\nContent-Length: " +
                                   std::to_string(body.size()) + "\r\nConnection: close\r\n\r\n" + body;
            for (std::size_t sent = 0; sent < response.size();) {
                ssize_t n = ::send(fd, response.data() + sent, response.size() - sent, MSG_NOSIGNAL);
                if (n <= 0) break;
                sent += static_cast<std::size_t>(n);
            }
            ::close(fd);
        }
    });
    return ntohs(addr.sin_port);
}

void MetricsHttpServer::stop() {
    if (!running_.exchange(false)) return;
    if (thread_.joinable()) thread_.join();
    ::close(listen_fd_);
    listen_fd_ = -1;
}

}
