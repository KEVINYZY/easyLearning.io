# Generates C++ RPC interfaces for express

import re
import copy
from utils import nested_dict, CodeTemplate, write
from gen_all import template_path,yprint

RPC_H = CodeTemplate.from_file(template_path + '/express_rpc.h')
RPC_CPP = CodeTemplate.from_file(template_path + '/express_rpc.cpp')

RPC_DECLARATION = CodeTemplate("""\
""")

RPC_DEFINE = CodeTemplate("""\
""")


def gen_rpcs(out, autograd_functions):
    pass
