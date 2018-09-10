# Generates C++ autograd functions APIs for express

import re
import copy
from utils import nested_dict, CodeTemplate, write
from gen_all import template_path,yprint

APIS_H = CodeTemplate.from_file(template_path + '/express_api.h')
APIS_CPP = CodeTemplate.from_file(template_path + '/express_api.cpp')

API_DECLARATION = CodeTemplate("""\
${api_return} ${api_name}(${api_args});
""")

API_DEFINE = CodeTemplate("""\
${api_return} ${api_name}(${api_args}) {
    ${api_return} result;
    
    auto op = std::make_shared<${op}>();
    ${op_inits}

    const_varptr_list  bottoms;
    ${bottoms_inits}

    auto ret = op->forward(bottoms);
    ${result_init}

    return result;
}

""")

def gen_apis(out, autograd_functions):
    
    all_apis_declarations = []
    all_apis_defines = []
    for func in autograd_functions:
        if ( "namespace"  in func["declaration"]["method_of"] and func["declaration"]["inplace"] == False):
            declr = gen_api_declaration(func)
            defn = gen_api_define(declr, func)

            all_apis_declarations.append(API_DECLARATION.substitute(declr))
            all_apis_defines.append( API_DEFINE.substitute(defn) )
        else:
            ## TODO
            pass

    top_env = {
        "auto_api_declarations" : all_apis_declarations,
        "auto_api_defines" : all_apis_defines,
    }

    write(out, "express_api.h", APIS_H, top_env)
    write(out, "express_api.cpp", APIS_CPP, top_env)

def gen_api_args(func, isDeclaration):
    declaration = func["declaration"]
    api_args = ""
    for i in range( len(declaration["formals"]) ):
        arg = declaration["formals"][i]
        if ( declaration["arguments"][i]["simple_type"] == "Tensor" ):
            arg = arg.replace("const Tensor &", "const_varptr")
            arg = arg.replace("Tensor", "variable")

        if ( declaration["arguments"][i]["simple_type"] == "TensorList" ):
            arg = arg.replace("TensorList", "const_varptr_list")

        if ( isDeclaration and  ("default" in declaration["arguments"][i] )):
            arg = arg + " = " + str(declaration["arguments"][i]["default"]).replace("False", "false").replace("True", "true")
        
        api_args = api_args + arg + ","
    api_args = api_args[:-1]
    return api_args


def gen_api_declaration(func):
    declaration = func["declaration"]
    env = {}
    env["api_name"] = declaration["api_name"];
    
    assert( len(declaration["formals"]) == len( declaration["arguments"]) )
    api_args = gen_api_args(func, True)
    
    env["api_args"] = api_args
    env["api_return"] = declaration["return_type"].replace("Tensor", "varptr")

    if ( env["api_return"].startswith("std::tuple") ):
        result_init = []
        for i in range( len ( declaration["returns"]) ):
            result_init.append( "std::get<{}>(result) = ret[{}];".format(i,i) )
        env["result_init"] = result_init
    elif env["api_return"].startswith("varptr") :
        env["result_init"] = [ "result = ret[0];" ]
    elif env["api_return"].startswith("std::vector"):
        env["result_init"] = [ "result = ret;" ]
    else:
        raise RuntimeError(
                "Unsupport return type : '{}' for op '{}' "
                .format(env["api_return"], func["op"] ) )
    
    return env

def gen_api_define(declr, func):
    declaration = func["declaration"]
    env = copy.copy(declr)
    
    api_args = gen_api_args(func, False)
    env["api_args"] = api_args
    env["op"] = func["op"]

    args_with_gradients = set()
    for arg in func["args_with_gradients"] :
        args_with_gradients.add( arg["name"] )

    ## op's parameter's init
    op_inits = []
    bottoms_inits = []
    tensorList_isinput = False
    for arg in declaration["arguments"]:
        if ( not arg["simple_type"].startswith("Tensor") ):
            op_init =  "op->{} = {};".format( arg["name"], arg["name"] )
            op_inits.append(op_init)
        else:
            if arg["name"] not in args_with_gradients:
                op_init =  "op->{} = {}->data();".format( arg["name"], arg["name"] )
                op_inits.append(op_init)
            else:
                if ( arg["simple_type"] == "Tensor" ):
                    bottoms_init = "bottoms.push_back({});".format( arg["name"] )
                    bottoms_inits.append( bottoms_init )
                elif ( arg["simple_type"] == "TensorList" ):
                    tensorList_isinput = True
                    bottoms_inits.append("bottoms = {};".format( arg["name"])) 
                else:
                    raise RuntimeError("op '{}' contains unsupported input : ".format( func["op"], arg["name"] )); 

    if ( tensorList_isinput and len(bottoms_inits) != 1) :
        raise RuntimeError("op '{}' TensorList must be only one tensor input!".format( func["op"] ))

    env["op_inits"] = op_inits
    env["bottoms_inits"] = bottoms_inits
    return env