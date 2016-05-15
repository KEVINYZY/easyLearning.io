require('nn')
require('optim')
require('xlua')

require('./boxLoss.lua')
local json = require "cjson"
local util = require "cjson.util"
local flags = require('./flags.lua')
local buildModel = require('./model.lua')
--local buildModel = require('./model_vgg.lua')
local buildData = require('./buildData.lua')

if ( flags._cuda_ == true) then
    require('cunn')
end

local g = {}
g.batchSize = 6

local doTrain = function()
    print(">>>>>>>>>>>>>TRAINING>>>>>>>>>>>>>");
    
    g.model:training()

    local parameters,gradParameters = g.model:getParameters()
    local batchInput, batchTarget = nil, nil
    local feval = function(x)
        -- get new parameters
        if x ~= parameters then
            parameters:copy(x)
        end
        
        -- reset gradients
        gradParameters:zero()
        
        local output = g.model:forward(batchInput)

        local f = g.criterion:forward(output, batchTarget)
        local df = g.criterion:backward(output, batchTarget)

        g.model:backward(batchInput, df)
        
        print(">>>>>>>>>>" .. f)
        return f, gradParameters
    end

    local index = 1
    local maxIterate = torch.floor( #g.trainSamples / g.batchSize )
    
    for i = 1, maxIterate do
        local batch = {}
        for j = index, index + g.batchSize - 1 do
            table.insert(batch, j)
        end
        batchInput, batchTarget = buildData(g.trainSamples, batch)

        g.optim(feval, parameters, g.optimState)
        --index = index + g.batchSize

        xlua.progress(i, maxIterate)
    end

end

local main = function()
    torch.setdefaulttensortype('torch.FloatTensor')
    
    g.model = buildModel()
    g.criterion = nn.BoxCriterion(1.0, flags)
    g.optim = optim.sgd
    g.optimState = {
        learningRate = 0.001
    }

    if ( flags._cuda_ ) then
        g.model:cuda()
        g.criterion:cuda()
    end

    -- building train and verify samples 
    g.allSamples = json.decode(util.file_load(flags.allDB))
    local samplesIndex = torch.randperm(#g.allSamples)
    g.trainSamples = {}
    g.verifySamples = {}
    for i = 1, math.floor(#g.allSamples * 0.8) do
        table.insert(g.trainSamples, g.allSamples[i])
    end
    for i = #g.trainSamples + 1, #g.allSamples do
        table.insert(g.verifySamples, g.allSamples[i])
    end
    for i = 1, 10 do
        doTrain()
    end
end 

main()

