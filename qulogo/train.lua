require('torch')
require('image')
local util = require('./util')
local data = require('./data')
local model = require('./model')

-- Checking input paramters and load config
local cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-d', 'sohu', 'The target dataset folder')
cmd:option('-neck', 384, 'The middle hidden vector')
cmd:option('-total_iterator', 100000, "Total iterate number")
cmd:option('-batch_size', 16, "Batch number")
cmd:option('-seed', 1979, "Random seed")

local opt = cmd:parse(arg)
local config = util.loadConfig(opt)
torch.manualSeed(opt.seed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- Checking model input and output size
--[[
local x = torch.rand(5, 3, config.inputWidth, config.inputHeight)
local gen = model.buildGenerator(opt, config)
local enc = model.buildEncoder(opt, config)
local disc = model.buildDiscriminator(opt, config)
local z = enc:forward(x)
local y = gen:forward(z)
local v = disc:forward(y)
print(z:size(), y:size(), v:size())
--]]

-- Global info
local inputBatch, centerBatch = nil, nil
local netG = nn.Sequential()
local netD = model.buildDiscriminator(opt, config)
netG:add( model.buildEncoder(opt, config) )
netG:add( model.buildGenerator(opt, config) )
netG:training()
netD:training()

local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

-- Training discriminator
local fDx = function(x) 
    gradParametersD:zero()
end

-- Traning genrator
local fGx = function(x)
    gradParametersG:zero()
end

local doTrain = function()
    for i = 1, opt.total_iterator do
        inputBatch, centerBatch = data.randomBatch()
         
                 
        collectgarbage() 
    end
end

