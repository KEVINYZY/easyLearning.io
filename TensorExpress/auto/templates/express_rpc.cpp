#include "generated/express_api.h"
#include "generated/express_operator.h"
#include "generated/express_rpc.h"

#include "express.h"

namespace express { namespace rpc {

${auto_rpc_defines}

int test(ExpressBackend* backend) {
    std::cout << " This is binded test" << std::endl;
    backend->test();
    return 0;
}

::rpc::server* bindRPCs(ExpressBackend* backend, unsigned int port) {
    ::rpc::server* srv = new ::rpc::server(port);

    ${auto_rpc_binds}

    return srv;
}

}} // namespace express::rpc
