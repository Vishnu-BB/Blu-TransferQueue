#include "transferqueue/GroupRouter.h"

#include <functional>
#include <stdexcept>

namespace tq {

GroupRouter::GroupRouter(int world_size) : world_size_(world_size) {
    if (world_size <= 0) {
        throw std::invalid_argument("GroupRouter: world_size must be > 0");
    }
}

int GroupRouter::target_rank(const std::string& group_id) const {
    std::size_t h = std::hash<std::string>{}(group_id);
    return static_cast<int>(h % static_cast<std::size_t>(world_size_));
}

}
