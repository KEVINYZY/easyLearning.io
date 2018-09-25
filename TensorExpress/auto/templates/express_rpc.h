#pragma once

#include <rpc/server.h>
#include <rpc/client.h>

struct ExpressBackend;

namespace express { namespace rpc {

using varptr = std::shared_ptr<Variable>;
using const_varptr = std::shared_ptr<const Variable>;
using varptr_list = std::vector< std::shared_ptr<Variable>>;
using const_varptr_list = std::vector< std::shared_ptr<const Variable>>;
using tensor_list = std::vector<at::Tensor>;

::rpc::server* bindRPCs(ExpressBackend* backend, unsigned int port);

}} // namespace express::rpc
