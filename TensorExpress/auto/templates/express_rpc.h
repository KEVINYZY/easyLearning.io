#pragma once

#include <rpc/server.h>
#include <rpc/client.h>

struct ExpressBackend;

namespace express { namespace rpc {

::rpc::server* bindRPCs(ExpressBackend* backend, unsigned int port);

}} // namespace express::rpc
