local ffi = require('ffi')
local C = transTorch._C

local toLinear = function(tm, caffeNet, layerName) 
    assert(tm.weight:type() == 'torch.FloatTensor')
    local weight = tm.weight:cdata()
    local bias = tm.bias:cdata()
    C.writeCaffeLinearLayer(caffeNet[0], layerName, weight, bias)
end

local toConv = function(tm, caffeNet, layerName)
    assert(tm.weight:type() == 'torch.FloatTensor')
    local weights = tm.weight:cdata()
    local bias = tm.bias:cdata()
    C.writeCaffeLinearLayer(caffeNet[0], layerName, weights, bias)
end

transTorch.loadCaffe = function(prototxt_name, binary_name, phase_name) 
    assert(type(prototxt_name) == 'string')
    assert(type(binary_name) == 'string')
    assert(type(phase_name) == 'string')
    
    local net = ffi.new("void*[1]")  
    net[0] = C.loadCaffeNet(prototxt_name, binary_name, phase_name)
    
    return net
end

transTorch.releaseCaffe = function(net) 
    C.releaseCaffeNet(net[0]);
end

transTorch.writeCaffe = function(net, fileName)
    C.saveCaffeNet(net[0], fileName);
end

transTorch.toCaffe = function(tmodel, caffeNet, layerName)
    local mtype = torch.type(tmodel)
    if ( mtype == 'nn.Linear' ) then
        toLinear(tmodel, caffeNet, layerName)
    elseif ( mtype == 'nn.BatchNormalization' or mtype == 'nn.SpatialBatchNormalization' ) then
        toBatchNorm(tmodel, caffeNet, layerName)
    elseif ( string.match(mtype, 'Convolution') ) then
        toConv(tmodel, caffeNet, layerName)
    end
end


