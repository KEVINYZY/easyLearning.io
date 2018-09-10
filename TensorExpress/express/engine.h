#pragma once

#include <vector>
#include <utility>

#include "variable.h"
#include "operator.h"

using at::Tensor;
using at::Scalar;
using at::IntList;
using at::TensorList;
using at::TensorGeometry;
using at::Generator;
using express::Variable;

namespace express {

using varptr = std::shared_ptr<Variable>;
using const_varptr = std::shared_ptr<const Variable>;
using varptr_list = std::vector< std::shared_ptr<Variable>>;
using const_varptr_list = std::vector< std::shared_ptr<const Variable>>;

int backward(varptr var, const_varptr grad);

};
