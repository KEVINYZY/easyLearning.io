from torch.autograd import Variable as V
from torch.autograd import Function as F

from . import pttoc

__all__ = ['buildGraph']

def _parseGraph(g, node, prev):
    if ( isinstance(node, V) ):
        if (node.requires_grad == True):
            g.addWeight(node)
        else:
            if ( not g.isInput(node) ):
                g.addConst(node)

    elif ( isinstance(node, F) ):
        if ( not g.hasNode(node) ):
            g.addOperator(node)
            if hasattr(node, 'previous_functions'):
                bottoms = []
                for u in node.previous_functions:
                    _parseGraph(g, u[0], node)
                    bottoms.append(u[0])

                g.setBottoms(node, bottoms)

    if ( prev != None):
        g.addTop(node, prev)

def buildGraph(inputs, outputs):
    graph = pttoc.ptGraph(inputs, outputs)
    for o in outputs:
        _parseGraph(graph, o.creator, None)
        graph.setBottoms(o, [o.creator])

    graph.reorg()
    return graph

