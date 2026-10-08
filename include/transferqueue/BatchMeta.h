#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tq {

using SampleId = std::uint64_t;

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
