#pragma once

#include <memory>
#include <string>

#include "transferqueue/Controller.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/StorageManager.h"

namespace tq {

// Low-level native API: wraps a Controller + StorageManager + Sampler.
// Mirrors upstream TransferQueueClient. Higher-level styles (Redis-like KV
// API, StreamingDataLoader) build on top of this -- not scaffolded yet,
// pending which style the rollout/trainer integration actually needs.
class TransferQueueClient {
public:
    TransferQueueClient(std::shared_ptr<TransferQueueController> controller,
                         std::shared_ptr<StorageManager> storage,
                         std::shared_ptr<BaseSampler> sampler);

    // Option B, upfront registration: declares this partition's field
    // dtypes once, typically before any put(). See Controller::declare_schema.
    void declare_schema(const std::string& partition_id, const std::unordered_map<std::string, FieldDtype>& schema);

    // Throws (via Controller::validate_schema) if any field in `data` has a
    // dtype that conflicts with a previously declared schema for
    // `partition_id`. No-op for fields never declared. `group_id`, if
    // given, is recorded for every sample in `data` -- GRPOGroupNSampler's
    // grouping key (see docs/GRPO_GROUP_N_SAMPLER.md). Every sample in one
    // put() call gets the same group_id; a caller whose ids span multiple
    // groups should call put() once per group.
    void put(const std::string& partition_id, const std::vector<std::string>& fields,
             const std::unordered_map<SampleId, Record>& data, const std::string& group_id = "");

    // Fetches up to batch_size samples ready for `task_name` (all `fields`
    // produced, not yet consumed by that task) and marks them consumed.
    std::unordered_map<SampleId, Record> get(const std::string& partition_id, const std::vector<std::string>& fields,
                                              const std::string& task_name, std::size_t batch_size);

private:
    std::shared_ptr<TransferQueueController> controller_;
    std::shared_ptr<StorageManager> storage_;
    std::shared_ptr<BaseSampler> sampler_;
};

}
