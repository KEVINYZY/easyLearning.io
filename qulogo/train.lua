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
cmd:option('-batch_size', 4, "Batch number")
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

-------------------------------
-- Global info
------------------------------
local netG = nn.Sequential()
local netD = model.buildDiscriminator(opt, config)
netG:add( model.buildEncoder(opt, config) )
netG:add( model.buildGenerator(opt, config) )
local parametersD, gradParametersD = netD:getParameters()
local parametersG, gradParametersG = netG:getParameters()

-- optim
local optimStateG = {
    learningRate = 0.001,
    beta1 = opt.beta1,
}
local optimStateD = {
    learningRate = 0.0001,
    beta1 = opt.beta1,
}
local optimG = optim.adam
local optimD = optim.adam

-- loss
local criterionD = nn.BCECriterion()
local criterionG = nn.MSECriterion()

-- input and output
local inputBatch, maskBatch = nil, nil
local label = torch.Tensor(opt.batch_size)
local gout = nil 

--------------------------------
-- Training discriminator
--------------------------------
local fDx = function(x) 
    netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    gradParametersD:zero()

    -- training with true
    label:fill(1);
    local output = netD:forward(maskBatch)
    local terror = criterionD:forward(output, label);
    local dfd = criterionD:backward(output, label);
    netD:backwoard(maskBatch, dfd)

    -- training with fake
    label:fill(0);
    gout = netG:forward(inputBatch)
    output = netD:fowrad(maskBatch)
    local ferror = criterionD:forward(output, label);
    local dfd = criterionD:backward(output, label);

    return terror+ferror, gradParametersD
end

--------------------------------
-- Traning genrator
--------------------------------
local fGx = function(x)
    gradParametersG:zero()
end

local doTrain = function()
    for i = 1, opt.total_iterator do
        netG:training()
        netD:training()
        inputBatch, maskBatch = data.randomBatch()
        
        collectgarbage() 
    end
end



