#include "transferqueue/StorageManager.h"

#include <algorithm>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <istream>
#include <memory>
#include <ostream>
#include <stdexcept>
#include <streambuf>
#include <unordered_set>

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
// Leading marker byte: OwnTensor has no zero-size tensors, so this keeps ""
// representable, and lets tensor_bytes() reject a tensor bytes_tensor() didn't write.
constexpr std::uint8_t kBytesMarker = 0xB7;
} // namespace

OwnTensor::Tensor bytes_tensor(std::string_view bytes) {
    OwnTensor::Tensor t(OwnTensor::Shape{{static_cast<std::int64_t>(bytes.size() + 1)}}, OwnTensor::Dtype::UInt8);
    auto* out = static_cast<std::uint8_t*>(t.data());
    out[0] = kBytesMarker;
    if (!bytes.empty()) {
        std::memcpy(out + 1, bytes.data(), bytes.size());
    }
    return t;
}

std::string tensor_bytes(const OwnTensor::Tensor& tensor) {
    if (tensor.dtype() != OwnTensor::Dtype::UInt8 || tensor.ndim() != 1 || tensor.numel() < 1 ||
        static_cast<const std::uint8_t*>(tensor.data())[0] != kBytesMarker) {
        throw std::invalid_argument("tensor_bytes: not a bytes field (expected a tensor written by bytes_tensor)");
    }
    const auto* data = static_cast<const char*>(tensor.data());
    return std::string(data + 1, data + tensor.nbytes());
}

namespace {

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

}

SimpleStorageManager::SimpleStorageManager(std::size_t capacity_bytes) : capacity_bytes_(capacity_bytes) {}

void SimpleStorageManager::put_data(const BatchMeta& meta, const std::unordered_map<SampleId, Record>& data) {
    put_impl(meta, data, /*wait=*/true);
}

bool SimpleStorageManager::try_put_data(const BatchMeta& meta, const std::unordered_map<SampleId, Record>& data) {
    return put_impl(meta, data, /*wait=*/false);
}

bool SimpleStorageManager::put_impl(const BatchMeta& meta, const std::unordered_map<SampleId, Record>& data,
                                    bool wait) {
    auto partition_of = partition_lookup(meta);

    // meta.fields is scanned once per (sample, field) pair below, twice
    // (byte-count pre-pass, then the write loop) -- an unordered_set turns
    // each check from O(meta.fields.size()) into O(1), worth it once a
    // batch has thousands of (sample, field) pairs (every real PUT).
    std::unordered_set<std::string> wanted_fields(meta.fields.begin(), meta.fields.end());

    std::size_t incoming_bytes = 0;
    for (const auto& [id, record] : data) {
        (void)id;
        for (const auto& [field, value] : record) {
            if (wanted_fields.count(field) != 0) {
                incoming_bytes += value.nbytes();
            }
        }
    }

    std::unique_lock<std::mutex> lock(mutex_);
    if (capacity_bytes_ > 0) {
        if (incoming_bytes > capacity_bytes_) {
            throw std::invalid_argument("SimpleStorageManager::put_data: batch of " +
                                         std::to_string(incoming_bytes) + " bytes exceeds total capacity of " +
                                         std::to_string(capacity_bytes_) + " bytes -- can never fit, even empty");
        }
        auto fits = [&] { return current_bytes_ + incoming_bytes <= capacity_bytes_; };
        if (!wait && !fits()) {
            return false;
        }
        space_available_.wait(lock, fits);
    }

    for (const auto& [id, record] : data) {
        auto& row = rows_[partition_of.at(id)][id];
        for (const auto& [field, value] : record) {
            if (wanted_fields.count(field) != 0) {
                auto existing = row.find(field);
                if (existing != row.end()) {
                    current_bytes_ -= existing->second.nbytes(); // overwriting, not adding
                }
                row[field] = value;
                current_bytes_ += value.nbytes();
            }
        }
    }
    return true;
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

// serialize_batch used to build its output via std::ostringstream (which
// grows and copies its internal buffer as it's written to), then copy that
// into a std::string (oss.str()), then copy THAT into the returned
// vector<uint8_t> -- three extra full-payload copies on top of the actual
// write. deserialize_batch had the same problem in reverse (copying the
// input vector into a std::string just to hand it to istringstream).
// Confirmed as a real, measured bottleneck -- see
// docs/TransferQueue-Benchmark.md's "Bottleneck 1"/"Bottleneck 2".
//
// Fix: these two streambufs let OwnTensor::save_tensor/load_tensor's
// existing istream&/ostream& API (unchanged, not touched here -- it's a
// separate library) read from and write to a vector<uint8_t>'s own memory
// directly, with no intermediate buffer at all. serialize_batch pre-sizes
// the output vector exactly once (via serialized_batch_size(), mirroring
// the write_* helpers' own byte-for-byte format below) instead of growing
// it, so there's exactly one allocation and zero buffer-growth copies.
//
// ponytail: the pre-sized vector<uint8_t> constructor still zero-fills its
// bytes before VectorWriteBuf overwrites them (one linear write-only pass,
// not a copy between two buffers) -- avoiding even that would need a
// custom allocator or C++23's resize_and_overwrite (this project targets
// C++20). Not one of the audit's confirmed copies; left as-is.
class VectorWriteBuf : public std::streambuf {
public:
    explicit VectorWriteBuf(std::vector<std::uint8_t>& out) {
        char* begin = reinterpret_cast<char*>(out.data());
        setp(begin, begin + out.size());
    }

protected:
    int_type overflow(int_type) override {
        // Only reachable if serialized_batch_size() under-counted the
        // buffer -- fail loudly instead of silently truncating output.
        throw std::logic_error("serialize_batch: pre-sized buffer was too small (size computation bug)");
    }
};

class VectorReadBuf : public std::streambuf {
public:
    VectorReadBuf(const std::uint8_t* data, std::size_t size) {
        char* begin = const_cast<char*>(reinterpret_cast<const char*>(data));
        setg(begin, begin, begin + size);
    }

};

std::size_t serialized_tensor_size(const OwnTensor::Tensor& tensor) {
    if (!tensor.is_valid()) {
        return 4; // "TNS0" -- see OwnTensor::save_tensor's invalid-tensor short form
    }
    return 4                                                            // "TNS1" magic
           + sizeof(int)                                                // dtype
           + sizeof(int)                                                // rank
           + static_cast<std::size_t>(tensor.ndim()) * sizeof(std::int64_t) // shape dims
           + tensor.nbytes();                                           // raw data
}

std::size_t serialized_batch_size(const std::unordered_map<SampleId, Record>& data) {
    std::size_t size = sizeof(std::uint32_t); // sample_count
    for (const auto& [id, record] : data) {
        (void)id;
        size += sizeof(SampleId) + sizeof(std::uint32_t); // id + field_count
        for (const auto& [field, tensor] : record) {
            size += sizeof(std::uint32_t) + field.size(); // write_string's length prefix + bytes
            size += serialized_tensor_size(tensor);
        }
    }
    return size;
}

} // namespace

std::vector<std::uint8_t> serialize_batch(const std::unordered_map<SampleId, Record>& data) {
    std::vector<std::uint8_t> out(serialized_batch_size(data));
    VectorWriteBuf buf(out);
    std::ostream os(&buf);
    write_u32(os, static_cast<std::uint32_t>(data.size()));
    for (const auto& [id, record] : data) {
        write_u64(os, id);
        write_u32(os, static_cast<std::uint32_t>(record.size()));
        for (const auto& [field, tensor] : record) {
            write_string(os, field);
            OwnTensor::save_tensor(tensor, os);
        }
    }
    return out;
}

std::unordered_map<SampleId, Record> deserialize_batch(std::vector<std::uint8_t> bytes) {
    // ponytail: OwnTensor::load_tensor allocates + copies each tensor (one
    // heap alloc per (sample, field)). Zero-copy views would need a
    // keep-alive view API in Tensor-Implementations; add it if this shows up
    // in the benchmark.
    VectorReadBuf buf(bytes.data(), bytes.size());
    std::istream is(&buf);

    std::unordered_map<SampleId, Record> data;
    std::uint32_t sample_count = read_u32(is);
    for (std::uint32_t i = 0; i < sample_count; ++i) {
        SampleId id = read_u64(is);
        std::uint32_t field_count = read_u32(is);
        Record record;
        for (std::uint32_t f = 0; f < field_count; ++f) {
            std::string field = read_string(is);
            record[field] = OwnTensor::load_tensor(is);
        }
        data[id] = std::move(record);
    }
    return data;
}

namespace {
constexpr char kShardMagic[4] = {'T', 'Q', 'S', 'H'};
constexpr std::uint32_t kShardFormat = 1;
} // namespace

void SimpleStorageManager::save(const std::string& path) const {
    // Serialize under the lock (consistent snapshot), write outside it.
    std::vector<std::pair<std::string, std::vector<std::uint8_t>>> partitions;
    {
        std::lock_guard<std::mutex> lock(mutex_);
        for (const auto& [partition_id, rows] : rows_) {
            partitions.emplace_back(partition_id, serialize_batch(rows));
        }
    }
    const std::string tmp = path + ".tmp";
    {
        std::ofstream os(tmp, std::ios::binary | std::ios::trunc);
        if (!os) throw std::runtime_error("SimpleStorageManager::save: cannot open " + tmp);
        os.write(kShardMagic, 4);
        os.write(reinterpret_cast<const char*>(&kShardFormat), sizeof(kShardFormat));
        auto count = static_cast<std::uint32_t>(partitions.size());
        os.write(reinterpret_cast<const char*>(&count), sizeof(count));
        for (const auto& [partition_id, bytes] : partitions) {
            auto id_len = static_cast<std::uint32_t>(partition_id.size());
            auto len = static_cast<std::uint64_t>(bytes.size());
            os.write(reinterpret_cast<const char*>(&id_len), sizeof(id_len));
            os.write(partition_id.data(), id_len);
            os.write(reinterpret_cast<const char*>(&len), sizeof(len));
            os.write(reinterpret_cast<const char*>(bytes.data()), static_cast<std::streamsize>(len));
        }
        if (!os.flush()) throw std::runtime_error("SimpleStorageManager::save: write failed for " + tmp);
    }
    std::filesystem::rename(tmp, path);
}

void SimpleStorageManager::load(const std::string& path) {
    std::ifstream is(path, std::ios::binary);
    if (!is) throw std::runtime_error("SimpleStorageManager::load: cannot open " + path);
    auto fail = [&] { throw std::runtime_error("SimpleStorageManager::load: corrupt or truncated " + path); };
    char magic[4];
    std::uint32_t format = 0, count = 0;
    is.read(magic, 4);
    is.read(reinterpret_cast<char*>(&format), sizeof(format));
    is.read(reinterpret_cast<char*>(&count), sizeof(count));
    if (!is || std::memcmp(magic, kShardMagic, 4) != 0 || format != kShardFormat) fail();

    std::unordered_map<std::string, std::unordered_map<SampleId, Record>> rows;
    std::size_t bytes_total = 0;
    for (std::uint32_t p = 0; p < count; ++p) {
        std::uint32_t id_len = 0;
        is.read(reinterpret_cast<char*>(&id_len), sizeof(id_len));
        std::string partition_id(id_len, '\0');
        is.read(partition_id.data(), id_len);
        std::uint64_t len = 0;
        is.read(reinterpret_cast<char*>(&len), sizeof(len));
        if (!is) fail();
        std::vector<std::uint8_t> bytes(len);
        is.read(reinterpret_cast<char*>(bytes.data()), static_cast<std::streamsize>(len));
        if (!is) fail();
        auto batch = deserialize_batch(std::move(bytes));
        for (const auto& [id, record] : batch) {
            for (const auto& [field, value] : record) {
                (void)field;
                bytes_total += value.nbytes();
            }
        }
        rows[partition_id] = std::move(batch);
    }

    std::lock_guard<std::mutex> lock(mutex_);
    rows_ = std::move(rows);
    current_bytes_ = bytes_total;
    space_available_.notify_all();
}

Tag SimpleStorageManager::stats() const {
    std::lock_guard<std::mutex> lock(mutex_);
    std::int64_t rows = 0;
    for (const auto& [partition_id, partition_rows] : rows_) {
        (void)partition_id;
        rows += static_cast<std::int64_t>(partition_rows.size());
    }
    return {{"bytes", static_cast<std::int64_t>(current_bytes_)},
            {"rows", rows},
            {"capacity_bytes", static_cast<std::int64_t>(capacity_bytes_)}};
}

Message make_batch_message(RequestType type, const std::string& sender_id, MessageBody body,
                            const std::unordered_map<SampleId, Record>& data,
                            std::optional<std::string> receiver_id) {
    body.payload = serialize_batch(data);
    return Message::create(type, sender_id, std::move(body), std::move(receiver_id));
}

std::unordered_map<SampleId, Record> extract_batch(Message message) {
    return deserialize_batch(std::move(message.body.payload));
}

}
