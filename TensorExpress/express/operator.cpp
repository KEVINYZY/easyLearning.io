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
        tops_.push_back( tops[i]->data() );
    }

    return tops;
}

tensor_list Operator::backward(const tensor_list& grads, std::vector<int> outputs) {
    assert( grads.size() == outputs.size() );
    assert( tops_.size() > 0);

    tensor_list all_grads;
    for(size_t i = 0; i < tops_.size(); i++) {
        bool find = false;
        for(size_t j = 0; j < outputs.size(); j++) {
            if ( i == outputs[j]) {
                all_grads.push_back( grads[j] );
                find = true;
                break;
            }
        }
        if (find == false) {
            auto zero_grad = at::zeros_like(tops_[i]);
            all_grads.push_back(zero_grad);
        }
    }
    assert( all_grads.size() == tops_.size());

    return _backward(all_grads);
}

//AccumulatedOperator
std::shared_ptr<AccumulatedOperator> AccumulatedOperator::singleton_;

varptr_list AccumulatedOperator::_forward(const const_varptr_list& bottoms) {
    varptr_list tops;
    return tops;
}

tensor_list AccumulatedOperator::_backward(const tensor_list& grads) {
    tensor_list bottom_grads;
    return bottom_grads;
}

} // namespace express
