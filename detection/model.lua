require 'nn'

local flags = require('./flags')

local function build_model()
    
    local model = nn.Sequential()
    
    -- 448 x 448
    model:add(nn.SpatialConvolution(3, 16, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2))
    
    model:add(nn.SpatialConvolution(32, 32, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(32, 64, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2))
 
    model:add(nn.SpatialConvolution(64, 64, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(64, 128, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2))
 
    model:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(128, 128, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(128, 256, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2))
    
    model:add(nn.SpatialConvolution(256, 256, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(256, 512, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialConvolution(512, 512, 3, 3, 1, 1))    
    model:add(nn.ReLU())
    model:add(nn.SpatialMaxPooling(2, 2))

    -- full connect
    model:add(nn.Reshape(512*8*8)) 
    model:add(nn.Linear(512*8*8, 1024))
    model:add(nn.ReLU())
    model:add(nn.Linear(1024, 4096))
    model:add(nn.ReLU())
    
    local mt = nn.ConcatTable()
    for i=1,flags.grid * flags.grid do
        local cc = nn.Sequential();
        cc:add( nn.Linear(4096, #flags.classmap + 1) )
        cc:add( nn.LogSoftMax() )
        mt:add(cc)
    end
    mt:add ( nn.Linear(4096, 4 * flags.grid * flags.grid) )
    model:add(mt)

    return model
end

return build_model

