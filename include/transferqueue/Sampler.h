#pragma once

#include <cstddef>
#include <cstdint>
#include <string>
#include <utility>
#include <vector>

#include "transferqueue/BatchMeta.h"

namespace tq {

struct StrandedGroup {
    std::string group_id;
    std::vector<SampleId> sample_ids; 
    std::int64_t oldest_age_ms;
};

class BaseSampler {
public:
    virtual ~BaseSampler() = default;

    virtual std::pair<std::vector<SampleId>, std::vector<SampleId>>
    sample(std::vector<SampleId> ready_ids, std::vector<std::string> group_ids, std::size_t batch_size) = 0;

    virtual std::vector<StrandedGroup> find_stranded(const std::vector<SampleId>& ready_ids,
                                                      const std::vector<std::string>& group_ids,
                                                      const std::vector<std::int64_t>& produced_at_ms,
                                                      std::int64_t now_ms, std::int64_t max_age_ms) const {
        (void)ready_ids;
        (void)group_ids;
        (void)produced_at_ms;
        (void)now_ms;
        (void)max_age_ms;
        return {};
    }
};
class FifoSampler : public BaseSampler {
public:
    std::pair<std::vector<SampleId>, std::vector<SampleId>> sample(std::vector<SampleId> ready_ids,
                                                                     std::vector<std::string> group_ids,
                                                                     std::size_t batch_size) override;
};

}
