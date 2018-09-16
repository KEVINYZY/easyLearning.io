#include "engine.h"

namespace express {

struct DAGNode {
public:
    varptr var;
    at::Tensor grad;
};

std::vector<DAGNode> doBackward(varptr var, at::Tensor grad) {
    if ( var->isConstant() ) {
        return {};
    }
    if ( var->isAccumulated() ) {
        assert(var->data().is_same_size(grad));

        auto oldGrad = var->grad();
        oldGrad = oldGrad + grad;

        return {};
    }
    
    // var is a true Variable
    var->grad() = grad;

    // call op's backward
    assert( var->data().is_same_size(grad) );
    tensor_list grads;
    std::vector<int> outputs;
    grads.push_back(grad);
    outputs.push_back(var->grad_output());

    auto backOp = var->grad_op();
    auto inputGrads = backOp->backward(grads, outputs);

    std::vector<DAGNode> ret;
    for(size_t i = 0; i < inputGrads.size(); i++) {
        DAGNode newNode;
        newNode.var = std::get<0>(inputGrads[i]);
        newNode.grad = std::get<1>(inputGrads[i]);

        ret.push_back(newNode);
    }
    return ret;
};

int backward(varptr var, const_varptr vgrad) {
    // create init DAG
    DAGNode newNode;
    newNode.var = var;
    newNode.grad = vgrad->data();
    std::list<DAGNode> allNodes = {newNode};

    while( allNodes.size() > 0) {
        auto v = allNodes.front().var;
        auto g = allNodes.front().grad;

        auto newNodes = doBackward(v, g);
        for(auto newNode : newNodes ) {
            allNodes.push_back(newNode);
        }
        allNodes.pop_front();
    }
    return 0;
}

}
