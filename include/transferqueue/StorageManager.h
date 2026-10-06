#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <optional>
#include <string>

#include "core/Tensor.h"
#include "transferqueue/BatchMeta.h"
#include "transferqueue/Message.h"

namespace tq {

using Record = std::unordered_map<std::string, OwnTensor::Tensor>;

FieldDtype to_field_dtype(OwnTensor::Dtype dtype);

class StorageManager {
public:
    virtual ~StorageManager() = default;

    virtual void put_data(const BatchMeta& meta, const std::unordered_map<SampleId, Record>& data) = 0;
    virtual std::unordered_map<SampleId, Record> get_data(const BatchMeta& meta) = 0;
    virtual void clear_data(const BatchMeta& meta) = 0;
};

class SimpleStorageManager : public StorageManager {
public:
    explicit SimpleStorageManager(std::size_t capacity_bytes = 0);

    void put_data(const BatchMeta& meta, const std::unordered_map<SampleId, Record>& data) override;
    std::unordered_map<SampleId, Record> get_data(const BatchMeta& meta) override;
    void clear_data(const BatchMeta& meta) override;

    std::size_t current_bytes() const;
    std::size_t capacity_bytes() const { return capacity_bytes_; }

private:
    std::size_t capacity_bytes_;
    std::size_t current_bytes_ = 0;
    std::unordered_map<std::string, std::unordered_map<SampleId, Record>> rows_;

    mutable std::mutex mutex_;
    std::condition_variable space_available_;
};

std::vector<std::uint8_t> serialize_batch(const std::unordered_map<SampleId, Record>& data);
std::unordered_map<SampleId, Record> deserialize_batch(const std::vector<std::uint8_t>& bytes);

Message make_batch_message(RequestType type, const std::string& sender_id, MessageBody body,
                            const std::unordered_map<SampleId, Record>& data,
                            std::optional<std::string> receiver_id = std::nullopt);
std::unordered_map<SampleId, Record> extract_batch(const Message& message);

}
