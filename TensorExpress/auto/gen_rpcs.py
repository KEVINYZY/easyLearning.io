# Generates C++ RPC interfaces for express

import re
import copy
from utils import nested_dict, CodeTemplate, write
from gen_all import template_path,yprint

RPC_H = CodeTemplate.from_file(template_path + '/express_rpc.h')
RPC_CPP = CodeTemplate.from_file(template_path + '/express_rpc.cpp')

RPC_DEFINE = CodeTemplate("""\

""")

RPC_BIND = CodeTemplate("""\
srv->bind("${api_name}", [&backend]
                        (const std::vector<std::string> outs, ${api_args})->int{
                            return test(backend);
                        });
""")

def gen_rpcs(out, autograd_functions):
    all_rpc_binds = []

    for func in autograd_functions:
        if ( "namespace"  in func["declaration"]["method_of"] and func["declaration"]["inplace"] == False):
            bind = gen_rpc_bind(func)
            all_rpc_binds.append(RPC_BIND.substitute(bind))
        elif ( "Tensor"  in func["declaration"]["method_of"] and func["declaration"]["inplace"] == True):
            pass
        else:
            pass


    top_env = {
        "auto_rpc_binds" : all_rpc_binds,
        "auto_rpc_defines" : []
    }

    write(out, "express_rpc.h", RPC_H, top_env)
    write(out, "express_rpc.cpp", RPC_CPP, top_env)

def gen_rpc_bind(func):
    declaration = func["declaration"]

    env = {}
    env["api_name"] = declaration["api_name"];

    api_args = ""
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
            arg = arg.replace("IntList", "const std::vector<int>&")

        if ( declaration["arguments"][i]["simple_type"] == "Scalar" ):
            arg = arg.replace("Scalar", "float")

        api_args = api_args + arg + ","
    api_args = api_args[:-1]
    env["api_args"] = api_args

    return env


