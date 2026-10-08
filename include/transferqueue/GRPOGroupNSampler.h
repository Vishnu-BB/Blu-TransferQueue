#pragma once

#include <cstddef>

#include "transferqueue/Sampler.h"

namespace tq {
class GRPOGroupNSampler : public BaseSampler {
public:
    explicit GRPOGroupNSampler(std::size_t n_samples_per_prompt = 1);

    std::pair<std::vector<SampleId>, std::vector<SampleId>> sample(std::vector<SampleId> ready_ids,
                                                                     std::vector<std::string> group_ids,
                                                                     std::size_t batch_size) override;

    std::vector<StrandedGroup> find_stranded(const std::vector<SampleId>& ready_ids,
                                              const std::vector<std::string>& group_ids,
                                              const std::vector<std::int64_t>& produced_at_ms, std::int64_t now_ms,
                                              std::int64_t max_age_ms) const override;

private:
    std::size_t n_samples_per_prompt_;
};

}
