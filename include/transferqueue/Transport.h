#pragma once

#include <memory>

#include "communication/include/ProcessGroupNCCL.h"

namespace tq {

class Transport {
public:
    explicit Transport(std::shared_ptr<ProcessGroupNCCL> pg);

private:
    std::shared_ptr<ProcessGroupNCCL> pg_;
};

}