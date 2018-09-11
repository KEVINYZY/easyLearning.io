#include "engine.h"

namespace express {

int backward(varptr var, const_varptr vgrad) {
    auto self = var->data();
    auto grad = vgrad->data();

    assert( self.is_same_size(grad) );

    auto backOp = var->grad_op();
    
    tensor_list grads;
    std::vector<int> outputs;
    grads.push_back(grad);
    outputs.push_back(var->grad_output());
    backOp->backward(grads, outputs);

    return 0;
}

}
