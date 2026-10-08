#pragma once

#include <string>

namespace tq {

class GroupRouter {
public:
    explicit GroupRouter(int world_size);

    int target_rank(const std::string& group_id) const;

private:
    int world_size_;
};

}
