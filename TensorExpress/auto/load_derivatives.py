# Parses derivatives.yaml into autograd functions
#
# Each autograd function is represented by dictionary containing a list of
# derivatives (also a dictionary). See `create_autograd_function` and
# `create_derivative` for the keys.
from collections import defaultdict
import copy
import re
import yaml
from utils import YamlLoader
from utils import IDENT_REGEX, split_name_params
from gen_all import yprint

# In principle this should live in derivatives.yaml, but I could not
# think of a good syntax for it
HARDCODED_DIFFERENTIABLE_OUTPUTS = {
    # Suppose that 'foo' is a function for which outputs 0 and 1 are
    # differentiable, and 2 is not.  Then you would write:
    # 'foo': (0, 1),
    '_cudnn_rnn': (0, 1, 2),
    # _cudnn_rnn outputs:
    #   0 => output
    #   1 => hy
    #   2 => cy
    #   3 => reserve
    #   4 => weight_buf
}

def load_derivatives(path, declarations):
    with open(path, 'r') as f:
        definitions = yaml.load(f, Loader=YamlLoader)

    declarations_by_signature = defaultdict(list)
    for declaration in declarations:
        declarations_by_signature[get_signature(declaration)].append(declaration)

    autograd_functions = [
        process_definition(defn, declarations_by_signature)
        for defn in definitions]

    ensure_unique_names(autograd_functions)
    match_declarations_with_autograd_functions(declarations, autograd_functions)

    return autograd_functions


# How do you feel about pasting declaration inside autograd function...
def create_autograd_function(name, derivatives, args_with_gradients, signature, declaration):
    op = to_camel_case(name) + 'Operator'
    op = op.replace('ForwardOperator', 'Operator')
    return {
        'name': name,
        'op': op,
        'declaration': declaration,
        'args_with_gradients': args_with_gradients,
        'signature': signature,
        'derivatives': derivatives,
    }


def create_derivative(declaration, formula, var_names):
    # Check that the referenced gradients in the formula are in bounds
    for i in used_gradient_indices(formula):
        if i >= len(declaration['returns']):
            raise RuntimeError(
                "Out of bounds grads access: derivative formula for {} "
                "used grads[{}], but the forward only returns {} outputs."
                .format(declaration['name'], i, len(declaration['returns'])))

    return {
        'formula': formula,
        'var_names': var_names,
    }


def process_definition(defn, declarations_by_signature):
    """Processes a single entry `defn` in derivatives.yaml"""

    def canonical_declaration(declarations, name):
        for declaration in declarations:
            if declaration['name'] == name:
                return declaration
        # some functions only have in-place variants
        assert name + '_' == declarations[0]['name']
        return declarations[0]

    def split_names(raw_names):
        """Given "foo, bar", return ["foo", "bar"]."""
        return [x.strip() for x in raw_names.split(',')]

    def check_grad_usage(defn_name, declaration, derivatives):
        """
        Check for some subtle mistakes one might make when writing gradients.
        These mistakes will compile, but will be latent until a function is
        used with double backwards.
        """

        used_grad = 0
        used_grads = 0
        fully_implemented = True
        used_grads_indices = []
        for d in derivatives:
            formula = d['formula']
            used_grad += len(re.findall(IDENT_REGEX.format('grad'), formula))
            used_grads += len(re.findall(IDENT_REGEX.format('grads'), formula))
            fully_implemented = \
                fully_implemented and \
                not re.search(IDENT_REGEX.format('not_implemented'), formula)
            used_grads_indices.extend(used_gradient_indices(formula))
        assert used_grads >= len(used_grads_indices)
        only_used_grads_indices = used_grads == len(used_grads_indices)

        if used_grad and used_grads:
            raise RuntimeError("Derivative definition of {} in derivatives.yaml illegally "
                               "mixes use of 'grad' and 'grads'. Consider replacing "
                               "occurrences of 'grad' with 'grads[0]'".format(defn_name))

        if only_used_grads_indices and set(used_grads_indices) == {0}:
            raise RuntimeError("Derivative definition of {} in derivatives.yaml solely "
                               "refers to 'grads[0]'.  If the first output is indeed the "
                               "only differentiable output, replace 'grads[0]' with 'grad'; "
                               "otherwise, there is a likely error in your derivatives "
                               "declaration.".format(defn_name))

        hardcoded_diff = HARDCODED_DIFFERENTIABLE_OUTPUTS.get(defn_name)
        if hardcoded_diff:
            if used_grad:
                raise RuntimeError("Derivative definition {} has hard-coded differentiable "
                                   "outputs in gen_autograd.py, but used grad (which implies "
                                   "only the first output is differentiable) in its "
                                   "derivative declaration.  You likely meant to write "
                                   "grads[i] for some i instead.".format(defn_name))
            if only_used_grads_indices and set(used_grads_indices) != set(hardcoded_diff):
                raise RuntimeError("Derivative definition {} has hard-coded differentiable "
                                   "outputs {}, but the used grads in the derivative "
                                   "definitions are only {}.  Either your derivatives "
                                   "declaration is wrong, or the value of "
                                   "HARDCODED_DIFFERENTIABLE_OUTPUTS in gen_autograd.py "
                                   "is wrong.".format(defn_name, hardcoded_diff,
                                                      used_grads_indices))
        else:
            if fully_implemented and not used_grad and \
               used_grads and only_used_grads_indices and \
               set(used_grads_indices) != set(range(len(declaration['returns']))):
                raise RuntimeError("Derivative definition of {} in derivatives.yaml does "
                                   "not refer to the gradients of all of its outputs.  Either "
                                   "the derivatives declaration is wrong, OR you have some "
                                   "non-differentiable outputs.  If you have a single "
                                   "differentiable output, make it the first output in ATen "
                                   "and reference its gradient with 'grad'; otherwise, hard "
                                   "code the list of differentiable outputs in "
                                   "HARDCODED_DIFFERENTIABLE_OUTPUTS in gen_autograd.py."
                                   .format(defn_name))

    def set_up_derivatives(defn_name, defn, declaration):
        # Determine the set of inputs which have gradients
        args_with_gradients_set = set()
        for raw_names in defn:
            args_with_gradients_set |= set(split_names(raw_names))

        # Next, let us determine the list of inputs in order.
        args_with_gradients = []
        for arg in declaration['arguments']:
            if arg['name'] not in args_with_gradients_set:
                continue
            args_with_gradients.append(arg)

        # Set up the derivative information
        derivatives = []
        for raw_names in sorted(defn.keys()):
            formula = defn[raw_names]
            names = split_names(raw_names)
            derivatives.append(create_derivative(declaration, formula, names))

        # Test to see if the use of 'grads' makes sense.
        check_grad_usage(defn_name, declaration, derivatives)

        return derivatives, args_with_gradients

    def unzip(xs):
        return zip(*xs)

    # NB: Removes 'name' from defn dictionary
    defn_name, params = split_name_params(defn.pop('name'))
    param_types, param_names = unzip([p.split(' ') for p in params if p != '*'])
    if 'grad_input_mask' in param_names:
        raise RuntimeError("Signature for {} has an argument named grad_input_mask, "
                           "but this name would be shadowed by our codegen. "
                           "Please use a different name in Declarations.cwrap."
                           .format(defn_name))
    signature = '{}({})'.format(defn_name, ', '.join(param_types))

    declarations = declarations_by_signature[signature]
    if len(declarations) == 0:
        avail = [k for k, v in declarations_by_signature.items()
                 if k.startswith(defn_name + '(') and len(v) > 0]
        raise RuntimeError('no ATen declaration found for: {}.  '
                           'Available signatures: {}'.format(signature, ', '.join(avail)))
    canonical = canonical_declaration(declarations, defn_name)

    # TODO: Check the types line up
    if len(param_names) != len(canonical['args']):
        raise RuntimeError('Signature for {} has {} arguments ({}), but '
                           'Declarations.yaml records {} arguments ({})'
                           .format(defn_name,
                                   len(param_names),
                                   ', '.join(param_names),
                                   len(canonical['args']),
                                   ', '.join(canonical['args'])))
    for i, (x, y) in enumerate(zip(param_names, canonical['args'])):
        if x != y:
            raise RuntimeError('Argument {} of {} has different names in '
                               'derivatives.yaml ({}) and '
                               'Declarations.yaml ({})'
                               .format(i, defn_name, x, y))

    derivatives, args_with_gradients = set_up_derivatives(defn_name, defn, canonical)
    return create_autograd_function(defn_name, derivatives, args_with_gradients, signature, canonical)


def ensure_unique_names(autograd_functions):
    # de-duplicate operation names
    # you end up with something like:
    #   AddBackward0
    #   AddBackward1
    # one for each overload
    functions_by_name = defaultdict(list)
    for func in autograd_functions:
        functions_by_name[func['op']].append(func)
    for op in functions_by_name.keys():
        overloads = functions_by_name[op]
        if len(overloads) > 1:
            for i, func in enumerate(overloads):
                func['op'] += str(i)


def get_signature(declaration, use_base_variant=False):
    name = declaration['name']
    arguments = declaration['arguments']
    if use_base_variant:
        if declaration['inplace']:
            assert name.endswith('_')
            name = name[:-1]
        elif name.endswith('_out'):
            name = name[:-4]
            arguments = [arg for arg in arguments if not arg.get('output', False)]
    simple_types = [arg['simple_type'] for arg in arguments]
    return '{}({})'.format(name, ', '.join(simple_types))


GRAD_INDEX_REGEX = r'(?:^|\W)grads\[(\d+)\]'


def used_gradient_indices(formula):
    """Determine a list of gradient indices (the i in grads[i]) that
    are used by the formula.

    >>> used_gradient_indices("foo(grads[0], grads[1])")
    [0, 1]
    """
    return [int(i) for i in re.findall(GRAD_INDEX_REGEX, formula)]


def to_camel_case(name):
    return ''.join([p.title() for p in name.split('_')])


def match_declarations_with_autograd_functions(declarations, autograd_functions):
    """Sets the "derivative" key on declarations to matching autograd functions

    In-place functions will use the out-of-place derivative definition if there
    is no in-place specific derivative.
    """

    functions_by_signature = {f['signature']: f for f in autograd_functions}

    def find_function(declaration):
        signature = get_signature(declaration)
        if signature in functions_by_signature:
            return functions_by_signature[signature]

        # if there is no exact match look for the out-of-place signature.
        # i.e mul() for mul_() or mul_out()
        signature = get_signature(declaration, use_base_variant=True)
        return functions_by_signature.get(signature)

    for declaration in declarations:
        declaration['derivative'] = find_function(declaration)
