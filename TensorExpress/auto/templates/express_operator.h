#pragma once

#include "variable.h"
#include "operator.h"

using at::Tensor;
using at::Scalar;
using at::IntList;
using at::TensorList;
using at::TensorGeometry;
using at::Generator;

using express::Variable;
using express::Operator;

namespace express { namespace op {

using varptr = std::shared_ptr<Variable>;
using const_varptr = std::shared_ptr<const Variable>;
using varptr_list = std::vector< std::shared_ptr<Variable>>;
using const_varptr_list = std::vector< std::shared_ptr<const Variable>>;
using tensor_list = std::vector<at::Tensor>;

${auto_operator_declarations}

}} // namespace express::op


