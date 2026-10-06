#pragma once

#include <cstdint>
#include <string>
#include <vector>

namespace tq {

using SampleId = std::uint64_t;

// Mirrors OwnTensor::Dtype (Tensor-Implementations/include/dtype/Dtype.h)
// exactly -- a total, lossless mapping, not a lossy subset -- so Controller
// (BluTrain-independent, Lane 1) can track and validate field dtypes
// without including core/Tensor.h. StorageManager.h (which already depends
// on OwnTensor) translates between the two at the boundary.
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

// Addresses a set of samples/fields for a Controller or StorageManager call.
// Mirrors upstream TransferQueue's BatchMeta: global_indexes -> sample_ids,
// partition_ids per sample (a batch can span partitions), and a
// production_status per sample already resolved for whichever fields this
// view was built for, with is_ready() derived from it.
//
// TODO: upstream's full field_schema (shape/nested-ness per field, beyond
// just dtype -- see FieldDtype above, added once there was a concrete need:
// Option B schema validation), custom_meta, and the lazy samples[] view are
// still skipped until there's a concrete need for them specifically.
struct BatchMeta {
    std::vector<SampleId> sample_ids;
    std::vector<std::string> partition_ids;   // one per sample_ids entry
    std::vector<std::string> fields;
    std::vector<bool> production_status;      // one per sample_ids entry

    BatchMeta() = default;

    // partition_ids must be the same length as sample_ids. production_status,
    // if given, must also match; if omitted, all samples default to
    // not-produced (matches upstream's default-zeros behavior).
    BatchMeta(std::vector<SampleId> ids, std::vector<std::string> partitions,
              std::vector<std::string> field_names = {},
              std::vector<bool> status = {});

    // True only if the batch is non-empty and every sample is produced.
    bool is_ready() const;
};

}
