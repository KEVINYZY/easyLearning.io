#include "engine.h"

namespace express {

int backward(varptr var, const_varptr vgrad) {
    auto self = var->data();
    auto grad = vgrad->data();

    assert( self.is_same_size(grad) );


    return 0;
}

}
