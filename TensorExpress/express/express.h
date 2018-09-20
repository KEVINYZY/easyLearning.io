#pragma once

#include <assert.h>
#include <map>
#include <string>

#include <rpc/server.h>
#include <rpc/client.h>

#include "variable.h"
#include "operator.h"
#include "engine.h"

using varptr = std::shared_ptr<express::Variable>;

namespace express { namespace rpc {

struct ExpressBackend {
public:
    ExpressBackend(unsigned int port);
    ~ExpressBackend();

    void run();

    void setVar(const std::string& name, varptr var) {
        vars_[name] = var;
    }
    void removeVar(const std::string& name) {
        vars_.erase(name);
    }
    varptr getVar(const std::string& name) {
        varptr ret;
        auto it = vars_.find(name);
        if (it != vars_.end()) {
            ret = it->second;
        }
        return ret;
    }


protected:
    ::rpc::server* rpcServer_;

    std::map<std::string, varptr> vars_;
};

}} // namespace express::rpc
