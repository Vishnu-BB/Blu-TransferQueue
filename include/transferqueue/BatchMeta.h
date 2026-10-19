#pragma once

#include <cstdint>
#include <limits>
#include <string>
#include <unordered_map>
#include <variant>
#include <vector>

namespace tq {

using SampleId = std::uint64_t;

// Returned by the KV lookups for a key that doesn't exist (and wasn't created).
inline constexpr SampleId kMissingSample = std::numeric_limits<SampleId>::max();

// KV-layer row metadata (verl's "tag"): status, global_steps, seq lens, ...
// Same variant RL's KVBatchMeta.h uses, so tags cross the boundary as-is.
using TagValue = std::variant<std::string, std::int64_t, double, bool>;
using Tag = std::unordered_map<std::string, TagValue>;

enum class FieldDtype : std::uint8_t {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Bfloat16,
    Float16,
    Float32,
    Float64,
    Bool,
    Complex32,
    Complex64,
    Complex128,
    Float4_e2m1,
    Float4_e2m1_2x,
};

struct BatchMeta {
    std::vector<SampleId> sample_ids;
    std::vector<std::string> partition_ids;   
    std::vector<std::string> fields;
    std::vector<bool> production_status;      

    BatchMeta() = default;

    BatchMeta(std::vector<SampleId> ids, std::vector<std::string> partitions,
              std::vector<std::string> field_names = {},
              std::vector<bool> status = {});

    bool is_ready() const;
};

}
