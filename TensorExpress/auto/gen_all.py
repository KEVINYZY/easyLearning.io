# gen_all.py generates C++ operators/apis for TensorExpress core.
#

import argparse
import copy
import os
import yaml
import json
from collections import defaultdict
from utils import YamlLoader

template_path = os.path.join(os.path.dirname(__file__), 'templates')
derivatives_path = os.path.join(os.path.dirname(__file__), 'derivatives.yaml')

def yprint(obj):
    print( yaml.dump(obj))

def format_return_type(returns):
    if len(returns) == 0:
        return 'void'
    elif len(returns) == 1:
        return returns[0]['type']
    else:
        return_types = [r['type'] for r in returns]
        return 'std::tuple<{}>'.format(','.join(return_types))

def load_aten_declarations(path):
    with open(path, 'r') as f:
        declarations = yaml.load(f, Loader=YamlLoader)

    # enrich declarations with additional information
    for declaration in declarations:
        for arg in declaration['arguments']:
            simple_type = arg['type']
            simple_type = simple_type.replace(' &', '').replace('const ', '')
            simple_type = simple_type.replace('Generator *', 'Generator')
            arg['simple_type'] = simple_type
        declaration['formals'] = [arg['type'] + ' ' + arg['name']
                                  for arg in declaration['arguments']]
        declaration['args'] = [arg['name'] for arg in declaration['arguments']]
        declaration['type_method_formals'] = [arg['type'] + ' ' + arg['name']
                                              for arg in declaration['arguments']
                                              if not arg.get('is_type_dispatched')]
        declaration['type_method_args'] = [arg['name'] for arg in declaration['arguments']
                                           if not arg.get('is_type_dispatched')]
        declaration['api_name'] = declaration['name']
        declaration['return_type'] = format_return_type(declaration['returns'])

        declaration['base_name'] = declaration['name']
    return declarations

def auto(aten_path, out):
    aten_decls = load_aten_declarations(aten_path)

    # Parse and load derivatives.yaml
    from load_derivatives import load_derivatives
    autograd_functions = load_derivatives(derivatives_path, aten_decls)

    from gen_apis import gen_apis
    gen_apis(out, autograd_functions)

    from gen_operators import gen_operators
    gen_operators(out, autograd_functions)

def main():
    parser = argparse.ArgumentParser(
        description='Generate operators/apis C++ files script')
    parser.add_argument('declarations', metavar='DECL',
                        help='path to Declarations.yaml')
    parser.add_argument('out', metavar='OUT',
                        help='path to output directory')
    args = parser.parse_args()
    auto(args.declarations, args.out)


if __name__ == '__main__':
    main()
