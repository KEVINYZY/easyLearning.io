require 'nn'
require 'cunn'
require('loadcaffe')

local flags = require('./flags')

string.startsWith = function(self, str) 
    return self:find('^' .. str) ~= nil
end

function loadVGG()
    local proto = './vgg19/VGG_ILSVRC_19_layers_deploy.prototxt'
    local caffeModel = './vgg19/VGG_ILSVRC_19_layers.caffemodel'

    local fullModel = loadcaffe.load(proto, caffeModel, 'nn')
    local cnn = nn.Sequential()
    for i = 1, #fullModel do
        local name = fullModel:get(i).name
        if ( name:startsWith('relu') or name:startsWith('conv') or name:startsWith('pool') ) then
            cnn:add( fullModel:get(i) )
        else
            break
        end
    end

    fullModel = nil
    collectgarbage()
    return cnn
end

local buildConv = function(input, output) 
    local conv = nn.SpatialConvolutionMM(input, output, 3, 3, 1, 1)
    local n = conv.kW * conv.kH * conv.nOutputPlane
    conv.weight:normal(0,math.sqrt(2/n))
    conv.bias:zero()

    return conv
end

local function build_model()
    local model = loadVGG()
    
    -- 512 x 14 x 14
    model:add(buildConv(512, 512))    
    model:add(nn.SpatialBatchNormalization(512, 1e-3))
    model:add(nn.ReLU())
    model:add(buildConv(512, 256))    
    model:add(nn.SpatialBatchNormalization(256, 1e-3))
    model:add(nn.ReLU())

    -- full connect
    model:add(nn.Reshape(256*10*10)) 
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(256*10*10, 1024))
    model:add(nn.BatchNormalization(1024))
    model:add(nn.ReLU())
    model:add(nn.Linear(1024, 4096))
    model:add(nn.BatchNormalization(4096))
    model:add(nn.ReLU())
    
    local mt = nn.ConcatTable()
    for i=1,flags.grid * flags.grid do
        local cc = nn.Sequential();
        cc:add( nn.Linear(4096, #flags.classmap + 1) )
        cc:add( nn.LogSoftMax() )
        mt:add(cc)
    end
    --mt:add ( nn.Linear(4096, 4 * flags.grid * flags.grid) )
    model:add(mt)

    return model
end

--[[
local model = build_model()
local img = torch.rand(1, 3, 448, 448)
x = model:forward(img)
print(x)
--]]

return build_model

