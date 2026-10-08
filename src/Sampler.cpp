#include "transferqueue/Sampler.h"

#include <algorithm>

namespace tq {

std::pair<std::vector<SampleId>, std::vector<SampleId>> FifoSampler::sample(std::vector<SampleId> ready_ids,
                                                                             std::vector<std::string> group_ids,
                                                                             std::size_t batch_size) {
    (void)group_ids;
    std::size_t n = std::min(batch_size, ready_ids.size());
    std::vector<SampleId> selected(ready_ids.begin(), ready_ids.begin() + n);
    std::vector<SampleId> remaining(ready_ids.begin() + n, ready_ids.end());
    return {selected, remaining};
}

}
