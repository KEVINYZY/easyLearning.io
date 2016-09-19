require('torch')
require('image')
require('nn')
require('optim')
require('xlua')

local util = require('./util')
local data = require('./data')
local model = require('./model')

-- Checking input paramters and load config
local cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-d', 'sohu', 'The target dataset folder')
cmd:option('-neck', 1024, 'The middle hidden vector')
cmd:option('-gpu', 1, 'Defaut using GPU 1')
cmd:option('-batch_size', 64, "Batch number")
cmd:option('-seed', 1979, "Random seed")
cmd:option('-threshold', 0.10, "Inpainting threshold")

local opt = cmd:parse(arg)
local config = util.loadConfig(opt)
if (opt.gpu ~= 0) then
    require('cunn')
end

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
local optimG = optim.adam
local optimD = optim.adam
local optimStateG = {
    learningRate = 0.001,
}
local optimStateD = {
    learningRate = 0.0001,
}

-- loss
local criterionD = nn.BCECriterion()
local criterionG = nn.MSECriterion()

-- input and output
local label = torch.Tensor(opt.batch_size)
local inputBatch, maskBatch = nil, nil
local outG, outD = nil, nil

--------------------------------
-- Training discriminator
--------------------------------
local fDx = function(x) 
    --netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    --netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    gradParametersD:zero()

    -- training with true
    label:fill(1)
    outD = netD:forward(maskBatch)
    local terror = criterionD:forward(outD, label)
    local dfd = criterionD:backward(outD, label)
    netD:backward(maskBatch, dfd)

    -- training with false
    label:fill(0)
    outG = netG:forward(inputBatch)
    outD = netD:forward(outG)
    local ferror = criterionD:forward(outD, label)
    dfd = criterionD:backward(outD, label)
    netD:backward(outG, dfd) 
   
    print("D error = " .. (terror + ferror) )
    return terror+ferror, gradParametersD
end

--------------------------------
-- Traning genrator
--------------------------------
local fGx = function(x)
    --netD:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    --netG:apply(function(m) if torch.type(m):find('Convolution') then m.bias:zero() end end)
    gradParametersG:zero()
   
    -- traninging with fake true
    label:fill(1)
    local ferror = criterionD:forward(outD, label)
    local dfd = criterionD:backward(outD, label)
    local dOutG = netD:updateGradInput(outG, dfd)
     
    local l2error = criterionG:forward(outG, maskBatch)
    local dfg = criterionG:backward(outG, maskBatch)
    
    dOutG:add(0.1, dfg) 
    netG:backward(inputBatch, dOutG)
   
    print("L2 error:" .. l2error .. " FG error:" .. ferror)
    return ferror + l2error, gradParametersG
end

local doTrain = function()
    if ( opt.gpu ~= 0) then
        netG:cuda()
        netD:cuda()
        criterionD:cuda()
        criterionG:cuda()
        label = label:cuda()
        parametersD, gradParametersD = netD:getParameters()
        parametersG, gradParametersG = netG:getParameters()
    end

    netG:training()
    netD:training()
    
    pageIndex = 1
    for i = 1, config.totalNumber do
        inputBatch, maskBatch = data.randomBatch(opt, config, pageIndex)
        if ( opt.gpu ~= 0) then
            inputBatch = inputBatch:cuda()
            maskBatch = maskBatch:cuda()
        end
     
        -- (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
        optimD(fDx, parametersD, optimStateD)

        -- (2) Update G network: maximize log(D(G(z)))
        optimG(fGx, parametersG, optimStateG)

        collectgarbage() 

        print("..................")
        --xlua.progress(i, opt.total_iterator)
    end
end

doTrain()

