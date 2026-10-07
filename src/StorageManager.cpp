#include "transferqueue/StorageManager.h"

#include <algorithm>
#include <sstream>
#include <stdexcept>

#include "core/Serialization.h"

namespace tq {

FieldDtype to_field_dtype(OwnTensor::Dtype dtype) {
    switch (dtype) {
        case OwnTensor::Dtype::Int8: return FieldDtype::Int8;
        case OwnTensor::Dtype::Int16: return FieldDtype::Int16;
        case OwnTensor::Dtype::Int32: return FieldDtype::Int32;
        case OwnTensor::Dtype::Int64: return FieldDtype::Int64;
        case OwnTensor::Dtype::UInt8: return FieldDtype::UInt8;
        case OwnTensor::Dtype::UInt16: return FieldDtype::UInt16;
        case OwnTensor::Dtype::UInt32: return FieldDtype::UInt32;
        case OwnTensor::Dtype::UInt64: return FieldDtype::UInt64;
        case OwnTensor::Dtype::Bfloat16: return FieldDtype::Bfloat16;
        case OwnTensor::Dtype::Float16: return FieldDtype::Float16;
        case OwnTensor::Dtype::Float32: return FieldDtype::Float32;
        case OwnTensor::Dtype::Float64: return FieldDtype::Float64;
        case OwnTensor::Dtype::Bool: return FieldDtype::Bool;
        case OwnTensor::Dtype::Complex32: return FieldDtype::Complex32;
        case OwnTensor::Dtype::Complex64: return FieldDtype::Complex64;
        case OwnTensor::Dtype::Complex128: return FieldDtype::Complex128;
        case OwnTensor::Dtype::Float4_e2m1: return FieldDtype::Float4_e2m1;
        case OwnTensor::Dtype::Float4_e2m1_2x: return FieldDtype::Float4_e2m1_2x;
    }
    throw std::invalid_argument("to_field_dtype: unknown OwnTensor::Dtype");
}

namespace {

// meta.partition_ids is parallel to meta.sample_ids (same convention
// BatchMeta's own constructor enforces). Builds a quick sample_id ->
// partition_id lookup; throws if the two vectors don't match length --
// loudly, rather than silently addressing the wrong (or no) partition.
std::unordered_map<SampleId, std::string> partition_lookup(const BatchMeta& meta) {
    if (meta.partition_ids.size() != meta.sample_ids.size()) {
        throw std::invalid_argument(
            "StorageManager: meta.partition_ids must be parallel to meta.sample_ids "
            "(sizes " +
            std::to_string(meta.partition_ids.size()) + " vs " + std::to_string(meta.sample_ids.size()) + ")");
    }
    std::unordered_map<SampleId, std::string> lookup;
    for (std::size_t i = 0; i < meta.sample_ids.size(); ++i) {
        lookup[meta.sample_ids[i]] = meta.partition_ids[i];
    }
    return lookup;
}

} // namespace

SimpleStorageManager::SimpleStorageManager(std::size_t capacity_bytes) : capacity_bytes_(capacity_bytes) {}

void SimpleStorageManager::put_data(const BatchMeta& meta, const std::unordered_map<SampleId, Record>& data) {
    auto partition_of = partition_lookup(meta);

    // Bytes this call would actually write (filtered by meta.fields, same
    // filter the write loop below applies) -- computed before locking since
    // it only reads the caller-supplied data, not shared state.
    std::size_t incoming_bytes = 0;
    for (const auto& [id, record] : data) {
        (void)id;
        for (const auto& [field, value] : record) {
            if (std::find(meta.fields.begin(), meta.fields.end(), field) != meta.fields.end()) {
                incoming_bytes += value.nbytes();
            }
        }
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (capacity_bytes_ > 0) {
        // A single batch whose own byte size already exceeds the total
        // capacity can NEVER satisfy the wait predicate below, even at
        // current_bytes_ == 0 -- waiting would block forever, not just
        // until space frees up. Reject immediately instead of deadlocking
        // the caller. Confirmed as a real bug via external review, not
        // assumed; see docs/UNIT_TEST_FINDINGS.md.
        if (incoming_bytes > capacity_bytes_) {
            throw std::invalid_argument("SimpleStorageManager::put_data: batch of " +
                                         std::to_string(incoming_bytes) + " bytes exceeds total capacity of " +
                                         std::to_string(capacity_bytes_) + " bytes -- can never fit, even empty");
        }
        space_available_.wait(lock, [&] { return current_bytes_ + incoming_bytes <= capacity_bytes_; });
    }

    for (const auto& [id, record] : data) {
        auto& row = rows_[partition_of.at(id)][id];
        for (const auto& [field, value] : record) {
            if (std::find(meta.fields.begin(), meta.fields.end(), field) != meta.fields.end()) {
                auto existing = row.find(field);
                if (existing != row.end()) {
                    current_bytes_ -= existing->second.nbytes(); // overwriting, not adding
                }
                row[field] = value;
                current_bytes_ += value.nbytes();
            }
        }
    }
}

std::unordered_map<SampleId, Record> SimpleStorageManager::get_data(const BatchMeta& meta) {
    auto partition_of = partition_lookup(meta);

    std::lock_guard<std::mutex> lock(mutex_);
    std::unordered_map<SampleId, Record> result;
    for (SampleId id : meta.sample_ids) {
        auto partition_it = rows_.find(partition_of.at(id));
        if (partition_it == rows_.end()) {
            continue;
        }
        auto it = partition_it->second.find(id);
        if (it == partition_it->second.end()) {
            continue;
        }

        // Filter by meta.fields, matching put_data's own filtering -- every
        // real caller (Client::get(), the colocated Server's and
        // StorageServer's GET_META/GET_DATA handlers) populates this with
        // the exact fields it wants back. An empty meta.fields means "no
        // filter requested" (defensive default, not currently exercised by
        // any real caller) and returns the whole row, same as before this
        // fix. See docs/UNIT_TEST_FINDINGS.md.
        if (meta.fields.empty()) {
            result[id] = it->second;
            continue;
        }
        Record filtered;
        for (const auto& field : meta.fields) {
            auto field_it = it->second.find(field);
            if (field_it != it->second.end()) {
                filtered[field] = field_it->second;
            }
        }
        result[id] = std::move(filtered);
    }
    return result;
}

std::size_t SimpleStorageManager::current_bytes() const {
    std::lock_guard<std::mutex> lock(mutex_);
    return current_bytes_;
}

void SimpleStorageManager::clear_data(const BatchMeta& meta) {
    auto partition_of = partition_lookup(meta);

    std::lock_guard<std::mutex> lock(mutex_);
    for (SampleId id : meta.sample_ids) {
        auto partition_it = rows_.find(partition_of.at(id));
        if (partition_it == rows_.end()) {
            continue;
        }
        auto it = partition_it->second.find(id);
        if (it == partition_it->second.end()) {
            continue;
        }
        for (const auto& [field, value] : it->second) {
            (void)field;
            current_bytes_ -= value.nbytes();
        }
        partition_it->second.erase(it);
        if (partition_it->second.empty()) {
            rows_.erase(partition_it);
        }
    }
    space_available_.notify_all();
}

namespace {

void write_u32(std::ostream& os, std::uint32_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }
void write_u64(std::ostream& os, std::uint64_t v) { os.write(reinterpret_cast<const char*>(&v), sizeof(v)); }

std::uint32_t read_u32(std::istream& is) {
    std::uint32_t v = 0;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!is) {
        throw std::invalid_argument("deserialize_batch: truncated stream");
    }
    return v;
}

std::uint64_t read_u64(std::istream& is) {
    std::uint64_t v = 0;
    is.read(reinterpret_cast<char*>(&v), sizeof(v));
    if (!is) {
        throw std::invalid_argument("deserialize_batch: truncated stream");
    }
    return v;
}

void write_string(std::ostream& os, const std::string& s) {
    write_u32(os, static_cast<std::uint32_t>(s.size()));
    os.write(s.data(), static_cast<std::streamsize>(s.size()));
}

std::string read_string(std::istream& is) {
    std::uint32_t len = read_u32(is);
    std::string s(len, '\0');
    is.read(s.data(), len);
    if (!is) {
        throw std::invalid_argument("deserialize_batch: truncated string");
    }
    return s;
}

} // namespace

std::vector<std::uint8_t> serialize_batch(const std::unordered_map<SampleId, Record>& data) {
    std::ostringstream oss(std::ios::binary);
    write_u32(oss, static_cast<std::uint32_t>(data.size()));
    for (const auto& [id, record] : data) {
        write_u64(oss, id);
        write_u32(oss, static_cast<std::uint32_t>(record.size()));
        for (const auto& [field, tensor] : record) {
            write_string(oss, field);
            OwnTensor::save_tensor(tensor, oss);
        }
    }
    std::string s = oss.str();
    return std::vector<std::uint8_t>(s.begin(), s.end());
}

std::unordered_map<SampleId, Record> deserialize_batch(const std::vector<std::uint8_t>& bytes) {
    std::istringstream iss(std::string(bytes.begin(), bytes.end()), std::ios::binary);

    std::unordered_map<SampleId, Record> data;
    std::uint32_t sample_count = read_u32(iss);
    for (std::uint32_t i = 0; i < sample_count; ++i) {
        SampleId id = read_u64(iss);
        std::uint32_t field_count = read_u32(iss);
        Record record;
        for (std::uint32_t f = 0; f < field_count; ++f) {
            std::string field = read_string(iss);
            record[field] = OwnTensor::load_tensor(iss);
        }
        data[id] = std::move(record);
    }
    return data;
}

Message make_batch_message(RequestType type, const std::string& sender_id, MessageBody body,
                            const std::unordered_map<SampleId, Record>& data,
                            std::optional<std::string> receiver_id) {
    body.payload = serialize_batch(data);
    return Message::create(type, sender_id, std::move(body), std::move(receiver_id));
}

std::unordered_map<SampleId, Record> extract_batch(const Message& message) {
    return deserialize_batch(message.body.payload);
}

}
