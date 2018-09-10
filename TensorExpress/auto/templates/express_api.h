#pragma once

#include "variable.h"

using at::Tensor;
using at::Scalar;
using at::IntList;
using at::TensorList;
using at::TensorGeometry;
using at::Generator;
using express::Variable;

namespace express { namespace api {

using varptr = std::shared_ptr<Variable>;
using const_varptr = std::shared_ptr<const Variable>;
using varptr_list = std::vector< std::shared_ptr<Variable>>;
using const_varptr_list = std::vector< std::shared_ptr<const Variable>>;

${auto_api_declarations}

}} // namespace express::api


