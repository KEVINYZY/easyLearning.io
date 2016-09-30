local ffi = require('ffi')
local C = transTorch._C

transTorch.loadCaffe = function(prototxt_name, binary_name, phase_name) 
    assert(type(prototxt_name) == 'string')
    assert(type(binary_name) == 'string')
    assert(type(phase_name) == 'string')
    
    local net = ffi.new("void*[1]")  
    net[0] = C.loadCaffeNet(prototxt_name, binary_name, phase_name)
    
    return net
end

transTorch.toCaffe = function(tmodel, caffeNet, layerName)
    if ( torch.type(tmodel) == 'nn.Linear' ) then
        toLinear(tmodel, caffeNet, layerName)
    end
end

transTorch.releaseCaffe = function(net) 
    C.releaseCaffeNet(net[0]);
end

local toLinear = function(tm, caffeNet, layerName) 
    assert(tm.weight:type() == 'torch.FloatTensor')
    local weights = tm.weight:cdata()
    local bias = tm.bias:cdata()
    C.writeCaffeLinearLayer(caffeNet[0], layerName, weights, bias)
end
