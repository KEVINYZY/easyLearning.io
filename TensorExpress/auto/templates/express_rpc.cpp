#include "generated/express_api.h"
#include "generated/express_operator.h"
#include "generated/express_rpc.h"

#include "express.h"

namespace express { namespace rpc {

${auto_rpc_defines}

void test(ExpressBackend* backend) {
    std::cout << " This is binded test" << std::endl;
    backend->test();
}

::rpc::server* bindRPCs(ExpressBackend* backend, unsigned int port) {
    ::rpc::server* srv = new ::rpc::server(port);

    srv->bind("Test", [&backend](){test(backend);});
    ${auto_rpc_binds}
    return srv;
}

}} // namespace express::rpc
