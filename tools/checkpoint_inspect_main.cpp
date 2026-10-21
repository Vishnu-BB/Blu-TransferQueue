// Human-readable dump of a TransferQueue checkpoint directory (what KVClient::save_checkpoint /
// rl::ITQClient::save_checkpoint writes): controller.tqck + shard-<i>.tqsh. Uses the same parsers a
// load does (TransferQueueController::parse_kv_checkpoint, SimpleStorageManager::read_shard_file),
// so what it prints is exactly what a restore would read. Read-only.
//
// Usage: checkpoint_inspect <checkpoint_dir> [max_rows_per_partition=5]
// Build: make checkpoint_inspect

#include <cstdint>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <map>
#include <set>
#include <string>
#include <variant>
#include <vector>

#include "transferqueue/Controller.h"
#include "transferqueue/StorageManager.h"

namespace fs = std::filesystem;
using OwnTensor::Dtype;

namespace {

const char* dtype_name(Dtype d) {
    switch (d) {
        case Dtype::Int8: return "int8";
        case Dtype::Int16: return "int16";
        case Dtype::Int32: return "int32";
        case Dtype::Int64: return "int64";
        case Dtype::UInt8: return "uint8";
        case Dtype::UInt16: return "uint16";
        case Dtype::UInt32: return "uint32";
        case Dtype::UInt64: return "uint64";
        case Dtype::Bfloat16: return "bf16";
        case Dtype::Float16: return "fp16";
        case Dtype::Float32: return "fp32";
        case Dtype::Float64: return "fp64";
        case Dtype::Bool: return "bool";
        default: return "other";
    }
}

std::string tag_value(const tq::TagValue& v) {
    if (auto* s = std::get_if<std::string>(&v)) return "\"" + *s + "\"";
    if (auto* i = std::get_if<std::int64_t>(&v)) return std::to_string(*i);
    if (auto* b = std::get_if<bool>(&v)) return *b ? "true" : "false";
    char buf[32];
    std::snprintf(buf, sizeof(buf), "%g", std::get<double>(v));
    return buf;
}

std::string tag_string(const tq::Tag& tag) {
    std::map<std::string, std::string> sorted;
    for (const auto& [k, v] : tag) sorted[k] = tag_value(v);
    std::string out;
    for (const auto& [k, v] : sorted) out += (out.empty() ? "" : ", ") + k + "=" + v;
    return "{" + out + "}";
}

// "int64[5] = [3, 1, 4, 1, 5]" or, for a bytes field, "bytes = \"...\"".
std::string tensor_preview(const OwnTensor::Tensor& t) {
    std::string shape;
    for (int64_t d = 0; d < t.ndim(); ++d) shape += (d ? "x" : "") + std::to_string(t.shape().dims[d]);
    if (t.dtype() == Dtype::UInt8 && t.ndim() == 1) {
        try {
            std::string text = tq::tensor_bytes(t);
            if (text.size() > 60) text = text.substr(0, 57) + "...";
            return "bytes[" + std::to_string(t.numel() - 1) + "] = \"" + text + "\"";
        } catch (const std::exception&) {
            // plain uint8 tensor, fall through
        }
    }
    std::string vals;
    const int64_t n = std::min<int64_t>(t.numel(), 6);
    for (int64_t i = 0; i < n; ++i) {
        char buf[32];
        const void* p = t.data();
        switch (t.dtype()) {
            case Dtype::Int64: std::snprintf(buf, sizeof(buf), "%lld", static_cast<long long>(static_cast<const std::int64_t*>(p)[i])); break;
            case Dtype::Int32: std::snprintf(buf, sizeof(buf), "%d", static_cast<const std::int32_t*>(p)[i]); break;
            case Dtype::Int16: std::snprintf(buf, sizeof(buf), "%d", static_cast<const std::int16_t*>(p)[i]); break;
            case Dtype::Int8: std::snprintf(buf, sizeof(buf), "%d", static_cast<const std::int8_t*>(p)[i]); break;
            case Dtype::UInt8: std::snprintf(buf, sizeof(buf), "%u", static_cast<const std::uint8_t*>(p)[i]); break;
            case Dtype::UInt16: std::snprintf(buf, sizeof(buf), "%u", static_cast<const std::uint16_t*>(p)[i]); break;
            case Dtype::Float32: std::snprintf(buf, sizeof(buf), "%g", static_cast<const float*>(p)[i]); break;
            case Dtype::Float64: std::snprintf(buf, sizeof(buf), "%g", static_cast<const double*>(p)[i]); break;
            case Dtype::Bool: std::snprintf(buf, sizeof(buf), "%s", static_cast<const std::uint8_t*>(p)[i] ? "true" : "false"); break;
            case Dtype::Bfloat16: {
                std::uint32_t bits = static_cast<std::uint32_t>(static_cast<const std::uint16_t*>(p)[i]) << 16;
                float f;
                std::memcpy(&f, &bits, sizeof(f));
                std::snprintf(buf, sizeof(buf), "%g", f);
                break;
            }
            default: std::snprintf(buf, sizeof(buf), "?"); break;
        }
        vals += (i ? ", " : "") + std::string(buf);
    }
    if (t.numel() > n) vals += ", ...";
    return std::string(dtype_name(t.dtype())) + "[" + shape + "] = [" + vals + "]";
}

struct KeyInfo {
    std::string key;
    tq::Tag tag;
    std::int32_t shard = -1;
    std::int64_t version = -1;
    std::vector<std::string> fields;
};

}  // namespace

int main(int argc, char** argv) {
    if (argc < 2) {
        std::fprintf(stderr, "usage: %s <checkpoint_dir> [max_rows_per_partition=5]\n", argv[0]);
        return 2;
    }
    const fs::path dir = argv[1];
    const int max_rows = argc > 2 ? std::stoi(argv[2]) : 5;

    std::printf("Checkpoint directory: %s\n", fs::absolute(dir).c_str());
    for (const auto& e : fs::directory_iterator(dir)) {
        std::printf("  %-20s %10ju bytes\n", e.path().filename().c_str(), static_cast<std::uintmax_t>(fs::file_size(e.path())));
    }

    // ---------------- controller.tqck ----------------
    const auto ckpt = tq::TransferQueueController::parse_kv_checkpoint((dir / "controller.tqck").string());
    std::string shards;
    for (auto s : ckpt.shard_indices) shards += (shards.empty() ? "" : ",") + std::to_string(s);
    std::printf("\n== controller.tqck (metadata)\n");
    std::printf("  policy version (global): %lld\n", static_cast<long long>(ckpt.version));
    std::printf("  saved from shards:       {%s}\n", shards.c_str());

    std::map<std::string, std::map<tq::SampleId, KeyInfo>> by_partition;  // partition -> id -> info
    for (const auto& rec : ckpt.records) {
        const auto& b = rec.body;
        for (size_t i = 0; i < b.sample_ids.size(); ++i) {
            KeyInfo info{b.keys[i], b.tags[i], b.sample_shard_indices[i], b.sample_versions[i], b.fields};
            by_partition[b.partition_id][b.sample_ids[i]] = std::move(info);
        }
    }
    for (const auto& [partition, keys] : by_partition) {
        int prompts = 0;
        for (const auto& [id, info] : keys) prompts += info.key.find('_') == std::string::npos;
        std::printf("\n  partition \"%s\": %zu key(s) -- %d prompt row(s) {uid}, %zu trajectory row(s) {uid}_{s}_{i}\n",
                    partition.c_str(), keys.size(), prompts, keys.size() - prompts);
        int shown = 0;
        for (const auto& [id, info] : keys) {
            if (shown++ == max_rows) {
                std::printf("    ... (%zu more; pass a larger max_rows to see them)\n", keys.size() - max_rows);
                break;
            }
            std::string fields;
            for (const auto& f : info.fields) fields += (fields.empty() ? "" : ",") + f;
            std::printf("    %-28s id=%-5llu shard=%-2d version=%-3lld fields=[%s]\n      tag %s\n", info.key.c_str(),
                        static_cast<unsigned long long>(id), info.shard, static_cast<long long>(info.version),
                        fields.c_str(), tag_string(info.tag).c_str());
        }
    }

    // ---------------- shard-<i>.tqsh ----------------
    int missing_data = 0, orphan_rows = 0;
    for (auto shard : ckpt.shard_indices) {
        const fs::path path = dir / ("shard-" + std::to_string(shard) + ".tqsh");
        const auto file = tq::SimpleStorageManager::read_shard_file(path.string());
        std::printf("\n== %s (tensor data for shard %d): %zu tensor bytes\n", path.filename().c_str(), shard, file.bytes);
        for (const auto& [partition, rows] : file.rows) {
            std::printf("  partition \"%s\": %zu row(s)\n", partition.c_str(), rows.size());
            int shown = 0;
            for (const auto& [id, record] : rows) {
                const auto& meta = by_partition[partition];
                auto it = meta.find(id);
                if (it == meta.end()) {
                    ++orphan_rows;
                    continue;
                }
                if (shown++ >= max_rows) continue;
                std::printf("    %s  (id=%llu)\n", it->second.key.c_str(), static_cast<unsigned long long>(id));
                std::map<std::string, std::string> sorted;
                for (const auto& [field, t] : record) sorted[field] = tensor_preview(t);
                for (const auto& [field, preview] : sorted) std::printf("      %-18s %s\n", field.c_str(), preview.c_str());
            }
            if (shown > max_rows) std::printf("    ... (%d more)\n", shown - max_rows);
        }
        // Every key the metadata places on this shard with produced fields must have them here.
        for (const auto& [partition, keys] : by_partition) {
            auto rows = file.rows.find(partition);
            for (const auto& [id, info] : keys) {
                if (info.shard != shard || info.fields.empty()) continue;
                const bool present = rows != file.rows.end() && rows->second.count(id);
                if (!present) ++missing_data;
                else
                    for (const auto& f : info.fields) missing_data += rows->second.at(id).count(f) == 0;
            }
        }
    }

    std::printf("\n== consistency\n");
    std::printf("  keys whose produced fields are missing from their shard: %d %s\n", missing_data,
                missing_data == 0 ? "(OK)" : "(BAD: restore would hand out rows without data)");
    std::printf("  shard rows not referenced by any key (harmless, e.g. written during a save): %d\n", orphan_rows);
    return missing_data == 0 ? 0 : 1;
}
