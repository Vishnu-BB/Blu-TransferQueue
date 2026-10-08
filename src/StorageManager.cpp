#include "transferqueue/StorageManager.h"

#include <algorithm>
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
        space_available_.wait(lock, [&] { return current_bytes_ + incoming_bytes <= capacity_bytes_; });
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

protected:
    // Needed for tellg()/seekg() -- load_tensor_view() below uses tellg()
    // to find each tensor's raw-data start offset (so it can view it in
    // place instead of reading/copying it) and seekg() to skip past it.
    pos_type seekoff(off_type off, std::ios_base::seekdir dir, std::ios_base::openmode) override {
        char* base = nullptr;
        switch (dir) {
            case std::ios_base::beg: base = eback() + off; break;
            case std::ios_base::cur: base = gptr() + off; break;
            case std::ios_base::end: base = egptr() + off; break;
            default: return pos_type(off_type(-1));
        }
        if (base < eback() || base > egptr()) {
            return pos_type(off_type(-1));
        }
        setg(eback(), base, egptr());
        return pos_type(base - eback());
    }

    pos_type seekpos(pos_type pos, std::ios_base::openmode which) override {
        return seekoff(off_type(pos), std::ios_base::beg, which);
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

// OwnTensor::load_tensor always allocates a fresh Tensor and copies the raw
// data into it -- for a batch of 9,216 (sample, field) tensors, that's
// 9,216 separate heap allocations plus 9,216 memcpy's, confirmed as a real,
// measured bottleneck (docs/TransferQueue-Benchmark.md's "Bottleneck 2"/
// "Bottleneck 5"). This is the zero-allocation counterpart: it parses the
// identical tensor header (duplicating OwnTensor::save_tensor's wire
// format -- magic/dtype/rank/dims -- on purpose, since Tensor::from_blob's
// view needs the exact byte offset the data starts at, which a
// general-purpose, stream-agnostic API like OwnTensor::load_tensor has no
// reason to expose) and then constructs a VIEW directly into `owner`'s own
// memory at the stream's current position instead of allocating and
// copying. `owner` is captured by the resulting Tensor's Storage (see
// Tensor::from_blob) as a keep-alive, so the one combined buffer
// deserialize_batch copies the wire bytes into outlives every tensor
// viewing into it for exactly as long as needed -- including well past
// this call, once stored into SimpleStorageManager::rows_.
OwnTensor::Tensor load_tensor_view(std::istream& is, const std::shared_ptr<std::vector<std::uint8_t>>& owner) {
    char magic[4];
    is.read(magic, 4);
    if (!is) {
        throw std::invalid_argument("deserialize_batch: truncated tensor header");
    }
    if (std::string(magic, 4) == "TNS0") {
        return OwnTensor::Tensor(); // invalid/default -- matches OwnTensor::load_tensor's own short form
    }
    if (std::string(magic, 4) != "TNS1") {
        throw std::invalid_argument("deserialize_batch: invalid tensor format");
    }

    int dtype_val = 0;
    is.read(reinterpret_cast<char*>(&dtype_val), sizeof(int));
    int rank = 0;
    is.read(reinterpret_cast<char*>(&rank), sizeof(int));
    if (!is || rank < 0) {
        throw std::invalid_argument("deserialize_batch: truncated tensor header");
    }
    auto dtype = static_cast<OwnTensor::Dtype>(dtype_val);

    std::vector<std::int64_t> dims(static_cast<std::size_t>(rank));
    for (auto& d : dims) {
        is.read(reinterpret_cast<char*>(&d), sizeof(std::int64_t));
    }
    if (!is) {
        throw std::invalid_argument("deserialize_batch: truncated tensor header");
    }
    OwnTensor::Shape shape({dims});

    std::size_t nbytes = OwnTensor::Tensor::dtype_size(dtype);
    for (std::int64_t d : dims) {
        nbytes *= static_cast<std::size_t>(d);
    }

    auto pos = static_cast<std::size_t>(is.tellg());
    if (pos + nbytes > owner->size()) {
        throw std::invalid_argument("deserialize_batch: truncated tensor data");
    }
    OwnTensor::Tensor tensor = OwnTensor::Tensor::from_blob(
        owner->data() + pos, shape, dtype, OwnTensor::DeviceIndex(OwnTensor::Device::CPU), owner);
    is.seekg(static_cast<std::streamoff>(pos + nbytes));
    if (!is) {
        throw std::invalid_argument("deserialize_batch: truncated tensor data");
    }
    return tensor;
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
    // `bytes` is already this function's own copy (or the caller's moved-in
    // buffer, paying zero extra copies -- see the declaration's comment).
    // Every view tensor below shares ownership of it, down from the old
    // per-tensor allocate-and-copy this replaces (9,216 of them, for a
    // 1024-sample x 9-field batch).
    auto owned = std::make_shared<std::vector<std::uint8_t>>(std::move(bytes));
    VectorReadBuf buf(owned->data(), owned->size());
    std::istream is(&buf);

    std::unordered_map<SampleId, Record> data;
    std::uint32_t sample_count = read_u32(is);
    for (std::uint32_t i = 0; i < sample_count; ++i) {
        SampleId id = read_u64(is);
        std::uint32_t field_count = read_u32(is);
        Record record;
        for (std::uint32_t f = 0; f < field_count; ++f) {
            std::string field = read_string(is);
            record[field] = load_tensor_view(is, owned);
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

std::unordered_map<SampleId, Record> extract_batch(Message message) {
    return deserialize_batch(std::move(message.body.payload));
}

}
