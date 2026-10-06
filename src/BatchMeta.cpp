#include "transferqueue/BatchMeta.h"

#include <stdexcept>

namespace tq {

BatchMeta::BatchMeta(std::vector<SampleId> ids, std::vector<std::string> partitions,
                      std::vector<std::string> field_names, std::vector<bool> status)
    : sample_ids(std::move(ids)),
      partition_ids(std::move(partitions)),
      fields(std::move(field_names)),
      production_status(std::move(status)) {
    if (partition_ids.size() != sample_ids.size()) {
        throw std::invalid_argument("BatchMeta: partition_ids length must match sample_ids");
    }
    if (production_status.empty()) {
        production_status.assign(sample_ids.size(), false);
    } else if (production_status.size() != sample_ids.size()) {
        throw std::invalid_argument("BatchMeta: production_status length must match sample_ids");
    }
}

bool BatchMeta::is_ready() const {
    if (sample_ids.empty()) {
        return false;
    }
    for (bool produced : production_status) {
        if (!produced) {
            return false;
        }
    }
    return true;
}

}
