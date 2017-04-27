import torch

import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function as F

from . import operators as OP

class ptObject(object):
    def __init__(self, type_, id_, obj_):
        self.id_ = id_
        self.type_ = type_
        self.obj_ = obj_
        self.name_ = None
        self.meta = {}

        self.bottoms = []
        self.tops = []

        if ( type_ == 'operator' ):
            self.op = OP.build(obj_)

        if ( type_ == 'input' or type_ == 'output'):
            self.size = [i for i in obj_.size()]

class ptForwarder(object):
    def __init__(self):
        pass

class ptGraph(object):
    def __init__(self, inputs, outputs):

        self.ids = {}
        self.inputs = []
        self.outputs = []

        self.forwards = []

        for i in inputs:
            pin = ptObject('input', id(i), i)
            pin.name_ = "input_{}".format(len(self.inputs))

            self.ids[id(i)] = pin
            self.inputs.append(pin)

        for o in outputs:
            pout = ptObject('output', id(o), o)
            pout.name_ = "output_{}".format(len(self.outputs))

            self.ids[id(o)] = pout
            self.outputs.append(pout)

    def isInput(self, node):
        if ( id(node) in self.ids ):
            p = self.ids[id(node)]
            if ( p.type_ == 'input' ):
                return True
            return False
        return False

    def hasNode(self, node):
        if ( id(node) in self.ids ):
            return True
        return False

    def addWeight(self, w):
        if ( id(w) in self.ids ):
            return

        pweight = ptObject('weight', id(w), w)
        pweight.name_ = "weight_{}".format( id(w) )

        self.ids[id(w)] = pweight

    def addConstant(self, c):
        if ( id(w) in self.ids ):
            return

        pconst = ptObject('constant', id(c), c)
        pconst.name_ = "constant_{}".format( id(c) )

        self.ids[id(c)] = pconst

    def addOperator(self, op):
        if ( id(op) in self.ids ):
            return

        pop = ptObject('operator', id(op), op)
        pop.name_ = "op_{}".format( id(op) )
        self.ids[id(op)] = pop

    def addTop(self, node, top):
        if ( id(node) in self.ids ):
            return
        if ( id(node) in self.ids ):
            return
        node = self.ids[id(node)]
        top = self.ids[id(top)]
        node.tops.append(top)

    def setBottoms(self, node, bottoms):
        node = self.ids[id(node)]
        for b in bottoms:
            b = self.ids[id(b)]
            node.bottoms.append(b)

    def _build(self, op, pos = None):
        if ( op.type_ == 'output' ):
            self.forwards.append(op)
            self._build(op.bottoms[0], len(self.forwards) - 1)
            return

        if ( op.type_ != "operator" ):
            return

        if ( op in self.forwards ):
            return

        self.forwards.insert(pos, op)
        for b in op.bottoms :
            self._build(b, pos)

    def reorg(self):
        for o in self.outputs:
            self._build(o)

        for i in range(len(self.forwards)):
            print( self.forwards[i].name_ )
