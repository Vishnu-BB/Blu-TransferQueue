#include "transferqueue/Transport.h"

namespace tq {

Transport::Transport(std::shared_ptr<ProcessGroupNCCL> pg) : pg_(std::move(pg)) {}

}
