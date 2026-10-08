#pragma once

#include <memory>
#include <string>

#include "transferqueue/Controller.h"
#include "transferqueue/Sampler.h"
#include "transferqueue/StorageManager.h"

namespace tq {

class TransferQueueClient {
public:
    TransferQueueClient(std::shared_ptr<TransferQueueController> controller,
                         std::shared_ptr<StorageManager> storage,
                         std::shared_ptr<BaseSampler> sampler);

    void declare_schema(const std::string& partition_id, const std::unordered_map<std::string, FieldDtype>& schema);

    void put(const std::string& partition_id, const std::vector<std::string>& fields,
             const std::unordered_map<SampleId, Record>& data, const std::string& group_id = "");

    std::unordered_map<SampleId, Record> get(const std::string& partition_id, const std::vector<std::string>& fields,
                                              const std::string& task_name, std::size_t batch_size);

private:
    std::shared_ptr<TransferQueueController> controller_;
    std::shared_ptr<StorageManager> storage_;
    std::shared_ptr<BaseSampler> sampler_;
};

}
