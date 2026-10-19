#pragma once

#include <condition_variable>
#include <cstdint>
#include <mutex>
#include <unordered_map>
#include <vector>

#include <optional>
#include <string>
#include <string_view>

#include "core/Tensor.h"
#include "transferqueue/BatchMeta.h"
#include "transferqueue/Message.h"

namespace tq {

using Record = std::unordered_map<std::string, OwnTensor::Tensor>;

FieldDtype to_field_dtype(OwnTensor::Dtype dtype);

// Non-tensor fields (verl's NonTensorData: strings, dicts like extra_fields /
// reward_extra_info). Stored as a 1-D UInt8 tensor (one marker byte + the raw
// bytes), so they ride
// through storage, the wire format and checkpoints like any other field. The
// caller picks the encoding (plain text, JSON, ...) and reads the field back
// with tensor_bytes(); nothing tags a field as "bytes", the field name does.
OwnTensor::Tensor bytes_tensor(std::string_view bytes);
std::string tensor_bytes(const OwnTensor::Tensor& tensor);  // throws unless written by bytes_tensor

class StorageManager {
public:
    virtual ~StorageManager() = default;

    virtual void put_data(const BatchMeta& meta, const std::unordered_map<SampleId, Record>& data) = 0;
    virtual std::unordered_map<SampleId, Record> get_data(const BatchMeta& meta) = 0;
    virtual void clear_data(const BatchMeta& meta) = 0;

    // Non-blocking put for the RPC servers: false (nothing written) when the
    // batch doesn't fit right now. A server must never wait for space inside
    // its worker pool -- the CLEAR that would free it could be queued behind.
    // Backends without a capacity limit never refuse.
    virtual bool try_put_data(const BatchMeta& meta, const std::unordered_map<SampleId, Record>& data) {
        put_data(meta, data);
        return true;
    }

    // Checkpoint: write every stored row to `path` / replace all rows with
    // the ones in `path`.
    virtual void save(const std::string& path) const = 0;
    virtual void load(const std::string& path) = 0;

    // Gauges for the metrics endpoint (e.g. bytes, rows).
    virtual Tag stats() const { return {}; }
};

class SimpleStorageManager : public StorageManager {
public:
    explicit SimpleStorageManager(std::size_t capacity_bytes = 0);

    // Blocks until the batch fits (in-process backpressure).
    void put_data(const BatchMeta& meta, const std::unordered_map<SampleId, Record>& data) override;
    bool try_put_data(const BatchMeta& meta, const std::unordered_map<SampleId, Record>& data) override;
    std::unordered_map<SampleId, Record> get_data(const BatchMeta& meta) override;
    void clear_data(const BatchMeta& meta) override;

    void save(const std::string& path) const override;
    void load(const std::string& path) override;
    Tag stats() const override;  // bytes, rows, capacity_bytes

    std::size_t current_bytes() const;
    std::size_t capacity_bytes() const { return capacity_bytes_; }

private:
    bool put_impl(const BatchMeta& meta, const std::unordered_map<SampleId, Record>& data, bool wait);

    std::size_t capacity_bytes_;
    std::size_t current_bytes_ = 0;
    std::unordered_map<std::string, std::unordered_map<SampleId, Record>> rows_;

    mutable std::mutex mutex_;
    std::condition_variable space_available_;
};

std::vector<std::uint8_t> serialize_batch(const std::unordered_map<SampleId, Record>& data);

// Takes `bytes` BY VALUE so a caller holding a disposable buffer can move
// it in. Every resulting tensor is an owned copy (OwnTensor::load_tensor).
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
