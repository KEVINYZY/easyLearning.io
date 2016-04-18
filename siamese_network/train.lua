require('nn')
require('optim')
require('xlua')

local buildModel = require('./model.lua')
local _CUDA_ = true
if ( _CUDA_ ) then
  require('cunn')
end
local g = {}
g.batchSize = 32


local loadBatch = function(index)
    local ii = index or 1
    
    local batch = {}
    batch.data = torch.Tensor(g.batchSize, 2, 1, 32, 32) 
    batch.label = torch.zeros(g.batchSize)
    for i = 1, g.batchSize do
        batch.data[i][1]:copy ( g.allTrainSamples.data[ii] )
        local iii = g.siamesePairs[ii]
        batch.data[i][2]:copy ( g.allTrainSamples.data[iii] )

        if ( g.allTrainSamples.labels[ii] == g.allTrainSamples.labels[iii] ) then
            batch.label[i] = 1
        end

        ii = i + 1
        if ( ii > g.trainNumber ) then
            ii = 1
        end

    end

    if _CUDA_ then
        batch.data = batch.data:cuda()
        batch.label = batch.label:cuda()
    end


    return batch, ii
end

local doTrain = function()
    if _CUDA_ then
        g.model:cuda()
        g.criterion:cuda()
    end
    g.model:training()

    local parameters,gradParameters = model:getParameters()
    local batch = nil
    local feval = function(x)
        -- get new parameters
        if x ~= parameters then
            parameters:copy(x)
        end
        
        -- reset gradients
        gradParameters:zero()
       
        local distance = g.model:forward(batch.data)
        
        local f = g.criterion:forward(distance, batch.label)
        local df = g.criterion:backward(distance, batch.label)
        
        g.model:backward(batch.data, df)

        print(">>>>>>>" .. f)
        return f, gradParameters
    end

    local index = 1
    local maxIterate = torch.floor( g.trainNumber / g.batchSize )
    for i = 1, maxIterate do
        batch, index = loadBatch(index)

        g.optim(feval, parameters, g.optimState)             
    
        collectgarbage();
        xlua.progress(i, maxIterate)
    end
end


local main = function()
    torch.setdefaulttensortype('torch.FloatTensor')

    g.model = buildModel()
    g.allTrainSamples = torch.load('./mnist.t7/train_32x32.t7', 'ascii')
    g.criterion = nn.HingeEmbeddingCriterion() 
    g.optim = optim.sgd
    g.optimState = {
        learningRate = 0.05
    }

    -- preprocessing train data
    local data = g.allTrainSamples.data
    g.allTrainSamples.data = data:type( torch.getdefaulttensortype() )
    data = g.allTrainSamples.data
    
    local std = data:std()
    local mean = data:mean()
    data:add(-mean);
    data:mul(1.0/std);

    local number = data:size(1)
    g.trainNumber = number
    for i = 1, 32 do
        g.siamesePairs = torch.randperm(number)
        doTrain()
    end
end 

main()

