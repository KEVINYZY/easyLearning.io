require 'nn'
require 'cunn'

local flags = require('./flags')

local function build_model()
    
    local model = nn.Sequential()
    
    -- 448 x 448
    model:add(nn.SpatialConvolutionMM(3, 16, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolutionMM(16, 32, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2))
   
    model:add(nn.SpatialConvolutionMM(32, 32, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolutionMM(32, 64, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2))
 
    model:add(nn.SpatialConvolutionMM(64, 64, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolutionMM(64, 128, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2))
 
    model:add(nn.SpatialConvolutionMM(128, 128, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolutionMM(128, 128, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolutionMM(128, 256, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2))
    
    model:add(nn.SpatialConvolutionMM(256, 256, 3, 3, 1, 1))    
    model:add(nn.SpatialBatchNormalization(256, 1e-3))
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolutionMM(256, 512, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolutionMM(512, 512, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2))

    -- full connect
    model:add(nn.Reshape(512*8*8)) 
    model:add(nn.Dropout(0.5))
    model:add(nn.Linear(512*8*8, 2048))
    model:add(nn.BatchNormalization(2048))
    model:add(nn.ReLU())
    model:add(nn.Linear(2048, 4096))
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

    -- initialization from MSR
    local function MSRinit(net)
    local function init(name)
        for k,v in pairs(net:findModules(name)) do
        local n = v.kW*v.kH*v.nOutputPlane
        v.weight:normal(0,math.sqrt(2/n))
        v.bias:zero()
        end
    end
    -- have to do for both backends
    init'nn.SpatialConvolutionMM'
    end

    MSRinit(model)


    return model
end

return build_model

