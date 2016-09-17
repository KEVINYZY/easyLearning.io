require('torch')
require('nn')

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local buildEncoder = function(opt, config)
    local nef = 48;
    local nw = config.inputWidth / 64 
    local nh = config.inputHeight / 64
    
    local netE = nn.Sequential()
    netE:add(SpatialConvolution(3, nef, 4, 4, 2, 2, 1, 1))
    netE:add(nn.LeakyReLU(0.2, true))
    
    -- form w*w --> w/32 * w/32: n, n*2, n*4, n*6, n*12 
    
    netE:add(SpatialConvolution(nef, nef, 4, 4, 2, 2, 1, 1))
    netE:add(SpatialBatchNormalization(nef)):add(nn.LeakyReLU(0.2, true))
    netE:add(SpatialConvolution(nef, nef * 2, 4, 4, 2, 2, 1, 1))
    netE:add(SpatialBatchNormalization(nef * 2)):add(nn.LeakyReLU(0.2, true))
    netE:add(SpatialConvolution(nef * 2, nef * 4, 4, 4, 2, 2, 1, 1))
    netE:add(SpatialBatchNormalization(nef * 4)):add(nn.LeakyReLU(0.2, true))
    netE:add(SpatialConvolution(nef * 4, nef * 6, 4, 4, 2, 2, 1, 1))
    netE:add(SpatialBatchNormalization(nef * 6)):add(nn.LeakyReLU(0.2, true))
    netE:add(SpatialConvolution(nef * 6, nef * 12, 4, 4, 2, 2, 1, 1))
    netE:add(SpatialBatchNormalization(nef * 12)):add(nn.LeakyReLU(0.2, true))
   
    netE:add(SpatialConvolution(nef * 12, opt.neck, nw, nh))
    
    return netE
end


local m = {}
m.buildEncoder = buildEncoder
return m
