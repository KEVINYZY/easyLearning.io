require('nn')
require('optim')
require('xlua')

local buildModel = require('./model.lua')
local _CUDA_ = true
if ( _CUDA_ ) then
  require('cunn')
end
local g = {}
g.batchSize = 128

local loadBatch = function(index, isTest)
    local ii = index or 1
   
    local data = g.allTrainSamples.data
    local labels = g.allTrainSamples.labels
    if ( isTest ) then
        data = g.allTestSamples.data
        labels = g.allTestSamples.labels
    end

    local number = data:size(1)

    local batch = {}
    batch.data = torch.Tensor(g.batchSize, 2, 1, 32, 32) 
    batch.label = torch.zeros(g.batchSize) - 1
    for i = 1, g.batchSize do
        batch.data[i][1]:copy ( data[ii] )
        local iii = g.siamese[ii]
        batch.data[i][2]:copy ( data[iii] )

        if ( labels[ii] == labels[iii] ) then
            batch.label[i] = 1
        end

        ii = ii + 1
        if ( ii > number ) then
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
    print(">>>>>>>>>>>>>TRAINING>>>>>>>>>>>>>");
    
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

        return f, gradParameters
    end

    local index = 1
    local maxIterate = torch.floor( g.allTrainSamples.data:size(1) / g.batchSize )
    for i = 1, maxIterate do
        batch, index = loadBatch(index)
        
        g.optim(feval, parameters, g.optimState)             
    
        collectgarbage();
        xlua.progress(i, maxIterate)
    end
end

local doTest = function()
    print(">>>>>>>>>>>>>TESTING>>>>>>>>>>>>>");

    if _CUDA_ then
        g.model:cuda()
        g.criterion:cuda()
    end
    g.model:evaluate()
    
    local index = 1
    local maxIterate = torch.floor( g.allTestSamples.data:size(1) / g.batchSize )
    local fsum = 0
    for i = 1, maxIterate do
        batch, index = loadBatch(index)
        
        local distance = g.model:forward(batch.data)
        local f = g.criterion:forward(distance, batch.label)
        fsum = fsum +f
        collectgarbage();
        xlua.progress(i, maxIterate)
    end

    print("Loss function = " .. fsum/maxIterate)
end

local main = function()
    torch.setdefaulttensortype('torch.FloatTensor')

    g.model = buildModel()
    g.allTrainSamples = torch.load('./mnist.t7/train_32x32.t7', 'ascii')
    g.allTestSamples = torch.load('./mnist.t7/test_32x32.t7', 'ascii')
    g.criterion = nn.HingeEmbeddingCriterion() 
    g.optim = optim.adam
    g.optimState = {
        learningRate = 0.001
    }

    -- preprocessing train data
    local data = g.allTrainSamples.data
    g.allTrainSamples.data = data:type( torch.getdefaulttensortype() )
    data = g.allTestSamples.data
    g.allTestSamples.data = data:type( torch.getdefaulttensortype() )

    data = g.allTrainSamples.data
    local std = data:std()
    local mean = data:mean()
    data:add(-mean);
    data:mul(1.0/std);
    data = g.allTestSamples.data
    data:add(-mean);
    data:mul(1.0/std);

    for i = 1, 32 do
        g.siamese = torch.randperm(g.allTrainSamples.data:size(1))
        doTrain()
        g.siamese = torch.randperm(g.allTestSamples.data:size(1))
        doTest()
    end
end 

main()

