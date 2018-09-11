# Generates C++ autograd functions operators for express

import re
import copy
from utils import nested_dict, CodeTemplate, write
from gen_all import template_path,yprint
from utils import IDENT_REGEX

OPERATORS_H = CodeTemplate.from_file(template_path + '/express_operator.h')
OPERATORS_CPP = CodeTemplate.from_file(template_path + '/express_operator.cpp')

OPERATOR_DECLARATION = CodeTemplate("""\
struct ${op_name} : public Operator {
public:
    ${op_name}() = default;
    virtual ~${op_name}() = default;

protected:
    virtual varptr_list _forward(const const_varptr_list& bottoms) override;
    virtual tensor_list _backward(const tensor_list& grads) override;

public:
    ${op_members}
};

""")

OPERATOR_DEFINE = CodeTemplate("""\
// ${op_name} implement
varptr_list  ${op_name}::_forward(const const_varptr_list& bottoms) {
    ${bottoms_inits}
    auto ret = ${call_api}
    
    varptr_list result;
    ${result_init}

    return result;
}
tensor_list ${op_name}::_backward(const tensor_list& grads) {
    assert( tops_.size() > 0);
    tensor_list ret;

    //saved inputs
    ${bottoms_inits_}    
    
    //output from tops
    ${tops_inits}

    //grad
    ${grad_init}

    //formula
    ${formulas}

    return ret;
}

""")


def gen_operators(out, autograd_functions):
    all_operators_declarations = []
    all_operators_defines = []
    for func in autograd_functions:
        if ( "namespace"  in func["declaration"]["method_of"] and func["declaration"]["inplace"] == False):
            declr = gen_operator_declaration(func)
            defn = gen_operator_define(func)

            all_operators_declarations.append(OPERATOR_DECLARATION.substitute(declr))
            all_operators_defines.append(OPERATOR_DEFINE.substitute(defn))
        else:
            ## TODO
            pass

    top_env = {
        "auto_operator_declarations" : all_operators_declarations,
        "auto_operator_defines" : all_operators_defines,
    }

    write(out, "express_operator.h", OPERATORS_H, top_env)
    write(out, "express_operator.cpp", OPERATORS_CPP, top_env)

def get_op_members(func):
    declaration = func["declaration"]

    args_with_gradients = set()
    for arg in func["args_with_gradients"] :
        args_with_gradients.add( arg["name"] )

    op_members = []
    for arg in declaration["arguments"]:
        if not arg["simple_type"].startswith("Tensor"): 
            op_member = "{} {};".format(arg["type"], arg["name"])
            op_members.append(op_member)
        elif ( arg["name"] not in args_with_gradients):
            op_member = "Tensor {};".format(arg["name"])
            op_members.append(op_member)

    return op_members

def gen_operator_declaration(func):
    env = {}
    env["op_name"] = func["op"]
    env["op_members"] = get_op_members(func)
    return env

def gen_operator_define(func):
    declaration = func["declaration"]
    env = {}
    env["op_name"] = func["op"]

    ## call api code
    call_api = "at::{}(".format( func["name"] )
    for arg in declaration["arguments"]:
        call_api = call_api + "{}, ".format(arg["name"])
    call_api = call_api[:-2] + ");"
    env["call_api"] = call_api

    args_with_gradients = set()
    for arg in func["args_with_gradients"] :
        args_with_gradients.add( arg["name"] )

    ## bottoms to input
    bottoms_inits = []
    tensorList_isinput = False
    for i in range( len(declaration["arguments"]) ):
        arg = declaration["arguments"][i]

        if not arg["simple_type"].startswith("Tensor"):
            continue
        if ( arg["name"] not in args_with_gradients ):
            continue

        if ( arg["simple_type"] == "Tensor" ):
            bottoms_init = "auto {} = bottoms[{}]->data();".format( arg["name"], len(bottoms_inits));
            bottoms_inits.append( bottoms_init )
        if ( arg["simple_type"] == "TensorList" ):
            tensorList_isinput = True
            bottoms_inits.append("std::vector<Tensor> {}_;".format( arg["name"] ))
            bottoms_inits.append("for(size_t i = 0; i < bottoms.size(); i++) {")
            bottoms_inits.append("   {}_.push_back(bottoms[i]->data());".format( arg["name"] ))
            bottoms_inits.append("}")
            bottoms_inits.append("TensorList {}({}_);".format( arg["name"], arg["name"]))
    
    if ( tensorList_isinput and len(bottoms_inits) != 5) :
        raise RuntimeError("op '{}' TensorList must be only one tensor input!".format( func["op"] ))
    
    env["bottoms_inits"] = bottoms_inits

    ## return to result
    api_return = declaration["return_type"].replace("Tensor", "varptr")
    if ( api_return.startswith("std::tuple") ):
        result_init = []
        for i in range( len ( declaration["returns"]) ):
            result_init.append( "result.push_back( express::make_variable( std::get<{}>(ret), get_shared_ptr(), {}) );".format(i, i) )
        env["result_init"] = result_init
    elif api_return.startswith("varptr") :
        env["result_init"] = [ "result.push_back( express::make_variable( ret, get_shared_ptr(), 0));" ]
    elif api_return.startswith("std::vector") :
        result_init = []
        result_init.append("for(size_t i = 0; i < ret.size(); i++) {")
        result_init.append("    result.push_back( express::make_variable( ret[i], get_shared_ptr(), i));")
        result_init.append("}")
        env["result_init"] = result_init
    else:
        raise RuntimeError(
                "Unsupport return type : '{}' for op '{}' "
                .format(api_return, func["op"] ) )

    ## backward implement
    gen_operator_backward(env, func)
    return env


def gen_operator_backward(env, func):
    declaration = func["declaration"]

    args_with_gradients = {}
    for arg in func["args_with_gradients"] :
        args_with_gradients[arg["name"]] = arg

    ## saved bottoms to input
    bottoms_inits = []
    tensorList_isinput = False
    for i in range( len(declaration["arguments"]) ):
        arg = declaration["arguments"][i]
        if not arg["simple_type"].startswith("Tensor"):
            continue
        if ( arg["name"] not in args_with_gradients ):
            continue

        if ( arg["simple_type"] == "Tensor" ):
            bottoms_init = "auto {} = bottoms_[{}]->data();".format( arg["name"], len(bottoms_inits) )
            bottoms_inits.append( bottoms_init )

        if ( arg["simple_type"] == "TensorList" ):
            tensorList_isinput = True
            bottoms_inits.append("std::vector<Tensor> {}_;".format( arg["name"] ))
            bottoms_inits.append("for(size_t i = 0; i < bottoms_.size(); i++) {")
            bottoms_inits.append("   {}_.push_back(bottoms_[i]->data());".format( arg["name"] ))
            bottoms_inits.append("}")
            bottoms_inits.append("TensorList {}({}_);".format( arg["name"], arg["name"]))

    if ( tensorList_isinput and len(bottoms_inits) != 5) :
        raise RuntimeError("op '{}' TensorList must be only one tensor input!".format( func["op"] ))

    env["bottoms_inits_"] = bottoms_inits
    
    ## saved tops to results
    tops_inits = []
    api_return = declaration["return_type"].replace("Tensor", "varptr")
    if ( api_return.startswith("std::tuple") or api_return.startswith("varptr") ):
        for i in range( len ( declaration["returns"]) ):
            tops_inits.append("auto {} = tops_[{}];".format( declaration["returns"][i]["name"], i));
    elif api_return.startswith("std::vector") :
        tops_inits.append("std::vector<Tensor> result;")
        tops_inits.append("for(size_t i = 0; i < tops.size(); i++) {")
        tops_inits.append("    result[i] = tops[i];")
        tops_inits.append("}")
    else:
        raise RuntimeError(
                "Unsupport return type : '{}' for op '{}' "
                .format(api_return, func["op"] ) )
    env["tops_inits"] = tops_inits

    ## grad and grads
    grad_init = []
    if api_return.startswith("varptr") or api_return.startswith("std::tuple"):
        grad_init.append("auto grad = grads[0];")

    env["grad_init"] = grad_init

    ## formula
    allFormula = {}
    formulas = []
    for formula in func["derivatives"] :
        if ( len(formula["var_names"]) == 1 ):
            v = formula["var_names"][0]
            f = formula["formula"]
            arg = args_with_gradients[v]

            if ( arg["simple_type"] == "Tensor" ):
                allFormula[v] = [ "auto {}_g = {};".format(v, f),
                              "ret.push_back({}_g);".format(v) ]
            elif (arg["simple_type"] == "TensorList") :
                assert( len(func["derivatives"]) == 1)
                codes = [ "auto {}_g = {};".format(v, f) ]
                codes.append("for(size_t i = 0; i < {}_g.size(); i++)".format(v))
                codes.append("{")
                codes.append("    ret.push_back({}_g[i]);".format(v))
                codes.append("}")
                allFormula[v] = codes
        else:
            f = formula["formula"]
            vn = len(formula["var_names"])
            if ( f.find("grad_input_mask") >= 0 ):
                formulas.append( "std::array<bool, {}> grad_input_mask = build_mask<{}>();".format(vn, vn))
            formulas.append( "auto tuple_g = {};".format(f) )
            for i in range(vn):
                v = formula["var_names"][i]
                codes = []
                codes.append("auto {}_g = std::get<{}>(tuple_g);".format(v, i))
                codes.append("ret.push_back({}_g);".format(v))
                allFormula[v] = codes

    for i in range( len(declaration["arguments"]) ):
        arg = declaration["arguments"][i]
        if not arg["simple_type"].startswith("Tensor"):
            continue
        if ( arg["name"] not in args_with_gradients ):
            continue
        v = arg["name"]
        if ( v in allFormula):
            formulas.extend(allFormula[v])

    env["formulas"] = formulas