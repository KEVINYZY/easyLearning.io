local ffi = require('ffi')
local C = transTorch._C


transTorch.loadCaffe = function(prototxt_name, binary_name, phase_name) 
    assert(type(prototxt_name) == 'string')
    assert(type(binary_name) == 'string')
    assert(type(phase_name) == 'string')
    
    local _net = ffi.new'void*[1]'  
    C.loadCaffe(_net, prototxt_name, binary_name, phase_name)
    
    return _net[0]
end


