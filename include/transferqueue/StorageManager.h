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

// Takes `bytes` BY VALUE: every resulting tensor is a zero-allocation VIEW
// into one internally-retained copy of it (see load_tensor_view() in
// StorageManager.cpp), not an owned, independently-allocated buffer -- so
// deserialize_batch needs to either copy or take ownership of its input
// somehow. By-value lets a caller that already holds a disposable buffer
// (e.g. extract_batch below, or a response Message about to go out of
// scope) move it in for free instead of paying a redundant copy on top of
// whatever copy already got the bytes there; a caller that only has a
// const reference still works exactly as before (one ordinary copy, same
// as this function always did internally).
std::unordered_map<SampleId, Record> deserialize_batch(std::vector<std::uint8_t> bytes);

Message make_batch_message(RequestType type, const std::string& sender_id, MessageBody body,
                            const std::unordered_map<SampleId, Record>& data,
                            std::optional<std::string> receiver_id = std::nullopt);

// Takes `message` BY VALUE for the same reason as deserialize_batch above --
// a caller holding a disposable Message (e.g. RpcClient.cpp's local
// `response`) can std::move() it in to avoid an extra payload copy; a
// caller with only a const Message& (e.g. Server.cpp's handle_request)
// still works, just copies the whole (typically payload-dominated) Message
// once, same as before.
std::unordered_map<SampleId, Record> extract_batch(Message message);

}
