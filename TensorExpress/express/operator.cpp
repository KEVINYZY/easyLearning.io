#include "operator.h"
#include "variable.h"

namespace express {

varptr_list Operator::forward(const const_varptr_list& bottoms) {
    bottoms_.clear();
    for(size_t i = 0; i < bottoms.size(); i++) {
        bottoms_.push_back( std::const_pointer_cast<Variable>(bottoms[i]) );
    }

    auto tops = _forward(bottoms);

    tops_.clear();
    for(size_t i = 0; i < tops.size(); i++) {
        tops_.push_back( tops[i] );
    }

    return tops;
}

tensor_list Operator::backward(const varptr_list& tops, const tensor_list& grads) {
    return _backward(tops, grads);
}

//AccumulatedOperator
std::shared_ptr<AccumulatedOperator> AccumulatedOperator::singleton_;

varptr_list AccumulatedOperator::_forward(const const_varptr_list& bottoms) {
    varptr_list tops;
    return tops;
}

tensor_list AccumulatedOperator::_backward(const varptr_list& tops, const tensor_list& grads) {
    tensor_list bottom_grads;
    return bottom_grads;
}

} // namespace express
