#include "generated/express_api.h"
#include "generated/express_operator.h"
#include "generated/express_rpc.h"

#include "express.h"

namespace express { namespace rpc {

${auto_rpc_defines}

::rpc::server* bindRPCs(ExpressBackend* backend, unsigned int port) {
    ::rpc::server* srv = new ::rpc::server(port);
    
    ${auto_rpc_binds}
    return srv;
}

}} // namespace express::rpc
