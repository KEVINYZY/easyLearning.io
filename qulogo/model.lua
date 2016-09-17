require('torch')
require('nn')

local SpatialBatchNormalization = nn.SpatialBatchNormalization
local SpatialConvolution = nn.SpatialConvolution
local SpatialFullConvolution = nn.SpatialFullConvolution

local function weights_init(m)
    local name = torch.type(m)
    if name:find('Convolution') then
        m.weight:normal(0.0, 0.02)
        m.bias:fill(0)
    elseif name:find('BatchNormalization') then
        if m.weight then m.weight:normal(1.0, 0.02) end
        if m.bias then m.bias:fill(0) end
    end
end
local buildEncoder = function(opt, config)
    local nef = 64
    local nw = config.inputWidth / 32 
    local nh = config.inputHeight / 32

    local netE = nn.Sequential()
    netE:add(SpatialConvolution(3, nef, 4, 4, 2, 2, 1, 1))
    netE:add(nn.LeakyReLU(0.2, true))

    netE:add(SpatialConvolution(nef, nef, 4, 4, 2, 2, 1, 1))
    netE:add(SpatialBatchNormalization(nef)):add(nn.LeakyReLU(0.2, true))
    netE:add(SpatialConvolution(nef, nef * 2, 4, 4, 2, 2, 1, 1))
    netE:add(SpatialBatchNormalization(nef * 2)):add(nn.LeakyReLU(0.2, true))
    netE:add(SpatialConvolution(nef * 2, nef * 4, 4, 4, 2, 2, 1, 1))
    netE:add(SpatialBatchNormalization(nef * 4)):add(nn.LeakyReLU(0.2, true))
    netE:add(SpatialConvolution(nef * 4, nef * 8, 4, 4, 2, 2, 1, 1))
    netE:add(SpatialBatchNormalization(nef * 8)):add(nn.LeakyReLU(0.2, true))

    netE:add(SpatialConvolution(nef * 8, opt.neck, nw, nh))

    netE:apply(weights_init)
    return netE
end

local buildGenerator = function(opt, config) 
    local nw = config.inputWidth / 32
    local nh = config.inputHeight / 32
    local ngf = 64

    local netG = nn.Sequential()

    netG:add(SpatialFullConvolution(opt.neck, ngf * 8, nw, nh))
    netG:add(SpatialBatchNormalization(ngf * 8)):add(nn.ReLU(true))

    netG:add(SpatialFullConvolution(ngf * 8, ngf * 4, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))
    netG:add(SpatialConvolution(ngf * 4, ngf * 4, 3, 3, 1, 1, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 4)):add(nn.ReLU(true))

    netG:add(SpatialFullConvolution(ngf*4, ngf*2, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf*2)):add(nn.ReLU(true))
    netG:add(SpatialConvolution(ngf * 2, ngf * 2, 3, 3, 1, 1, 1, 1))
    netG:add(SpatialBatchNormalization(ngf * 2)):add(nn.ReLU(true))

    netG:add(SpatialFullConvolution(ngf*2, ngf, 4, 4, 2, 2, 1, 1))
    netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))
    netG:add(SpatialConvolution(ngf, ngf, 3, 3, 1, 1, 1, 1))
    netG:add(SpatialBatchNormalization(ngf)):add(nn.ReLU(true))

    netG:add(SpatialConvolution(ngf, 3, 3, 3, 1, 1, 1, 1))
    netG:add(nn.Tanh())

    netG:apply(weights_init)
    return netG
end

local buildDiscriminator = function(opt, config) 
    local ndf = 64
    local nw = config.inputWidth / 32
    local nh = config.inputHeight / 32

    local netD = nn.Sequential()
    netD:add(SpatialConvolution(3, ndf,  4, 4, 2, 2, 1, 1))
    netD:add(nn.LeakyReLU(0.2, true))

    netD:add(SpatialConvolution(ndf, ndf * 4, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    netD:add(SpatialConvolution(ndf * 4, ndf * 4, 3, 3, 1, 1, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))

    netD:add(SpatialConvolution(ndf * 4, ndf * 4, 4, 4, 2, 2, 1, 1))
    netD:add(SpatialBatchNormalization(ndf * 4)):add(nn.LeakyReLU(0.2, true))
    netD:add(SpatialConvolution(ndf * 4, ndf * 8, 3, 3, 1, 1, 1, 1))
    netD:add(SpatialBatchNormalization(ndf*8)):add(nn.ReLU(true))

    netD:add(SpatialConvolution(ndf * 8, 1, nw, nh))
    netD:add(nn.Sigmoid())
    netD:add(nn.View(1):setNumInputDims(3))
    
    netD:apply(weights_init) 
    return netD
end

local m = {}
m.buildEncoder = buildEncoder
m.buildGenerator = buildGenerator
m.buildDiscriminator = buildDiscriminator
return m
