# Generates C++ RPC interfaces for express

import re
import copy
from utils import nested_dict, CodeTemplate, write
from gen_all import template_path,yprint

RPC_H = CodeTemplate.from_file(template_path + '/express_rpc.h')
RPC_CPP = CodeTemplate.from_file(template_path + '/express_rpc.cpp')

RPC_DEFINE = CodeTemplate("""\
int ${api_name}(ExpressBackend* backend, const std::vector<std::string> outs, ${api_args}) {

    ${arg_to_varptr}

    auto ret = ::express::api::${real_api_name}(${call_seq});

    ${ret_to_outs}

    return 0;
}
""")

RPC_BIND = CodeTemplate("""\
srv->bind("${api_name}", [&backend]
                        (const std::vector<std::string> outs, ${api_args})->int{
                            return ${api_name}(backend, outs, ${args_name});
                        });
""")

def gen_rpcs(out, autograd_functions):
    all_rpc_defines = []
    all_rpc_binds = []

    all_names = {}

    for func in autograd_functions:
        if ( "namespace"  in func["declaration"]["method_of"] and func["declaration"]["inplace"] == False):
            name = func["declaration"]["api_name"]
            if ( name not in all_names ):
                all_names[name] = 0
            else:
                all_names[name] = all_names[name] + 1
                name = name + str(all_names[name])
            bind = gen_rpc_bind(func)
            bind["api_name"] = name
            all_rpc_binds.append(RPC_BIND.substitute(bind))

            define = gen_rpc_define(func)
            define["api_name"] = name
            all_rpc_defines.append(RPC_DEFINE.substitute(define))

        elif ( "Tensor"  in func["declaration"]["method_of"] and func["declaration"]["inplace"] == True):
            ## TODO
            pass
        else:
            pass


    top_env = {
        "auto_rpc_binds" : all_rpc_binds,
        "auto_rpc_defines" : all_rpc_defines
    }

    write(out, "express_rpc.h", RPC_H, top_env)
    write(out, "express_rpc.cpp", RPC_CPP, top_env)

def gen_rpc_define(func):
    declaration = func["declaration"]

    env = gen_rpc_bind(func)

    arg_to_varptr = []
    call_seq = ""
    for i in range( len(declaration["formals"]) ):
        arg = declaration["args"][i]
        if ("default" in declaration["arguments"][i]):
            continue

        arg_name = declaration["args"][i]
        if ( declaration["arguments"][i]["simple_type"] == "Tensor" ):
            arg_name = arg_name + "_"

            varptr = 'auto {}_ = backend->getVar({});'.format(arg, arg)
            arg_to_varptr.append(varptr)
            
        if ( declaration["arguments"][i]["simple_type"] == "TensorList" ):
            arg_name = arg_name + "_"

            varptr = 'const_varptr_list {}_;\n'.format( declaration["args"][i] )
            varptr = varptr + 'for (auto &i : {}) {}_.push_back( backend->getVar(i) );\n'.format( arg, arg )
            arg_to_varptr.append(varptr)

        call_seq = call_seq + arg_name + ","

    ret_to_outs = []
    api_return = declaration["return_type"].replace("Tensor", "varptr")
    if ( api_return.startswith("std::tuple") ):
        for i in range( len ( declaration["returns"]) ):
            ret_to_outs.append("backend->setVar(outs[{}], std::get<{}>(ret));".format(i,i) )
    elif api_return.startswith("varptr") :
        ret_to_outs.append("backend->setVar(outs[0], ret);")
    else:
        raise RuntimeError(
                "Unsupport return type : '{}' for op '{}' "
                .format(env["api_return"], func["op"] ) )

    env["real_api_name"] = func["declaration"]["api_name"]
    env["arg_to_varptr"] = arg_to_varptr
    env["call_seq"] = call_seq[:-1]
    env["ret_to_outs"] = ret_to_outs

    return env

def gen_rpc_bind(func):
    declaration = func["declaration"]
    env = {}

    api_args = ""
    args_name = ""
    for i in range( len(declaration["formals"]) ):
        arg = declaration["formals"][i]
        if ("default" in declaration["arguments"][i]):
            continue

        if ( declaration["arguments"][i]["simple_type"] == "Tensor" ):
            arg = arg.replace("const Tensor &", "const std::string&")
            arg = arg.replace("Tensor", "const std::string&")

        if ( declaration["arguments"][i]["simple_type"] == "TensorList" ):
            arg = arg.replace("TensorList", "const std::vector<std::string>&")

        if ( declaration["arguments"][i]["simple_type"] == "IntList" ):
            arg = arg.replace("IntList", "const std::vector<int64_t>&")

        if ( declaration["arguments"][i]["simple_type"] == "Scalar" ):
            arg = arg.replace("Scalar", "float")

        args_name = args_name + declaration["args"][i] + ","
        api_args = api_args + arg + ","

    api_args = api_args[:-1]
    args_name = args_name[:-1]

    env["api_args"] = api_args
    env["args_name"] = args_name

    return env