require('cunn')
require('optim')
require('xlua')

-- init
torch.setdefaulttensortype('torch.FloatTensor')
local DataLoader = paths.dofile('data_loader.lua')

-- global
g = {}

_ = require('./model.lua')
g.fixedCNN = _.fixedCNN
g.featureCNN = _.featureCNN
g.lossLayers = _.lossLayers
g.modelInfo = _.info

g.dataLoader = DataLoader.new(4, 'data/allSamples.t7', g.modelInfo)

local doTrain = function(itnum)
    print(">>>>>>>>>>>>>TRAINING>>>>>>>>>>>>>");

    g.featureCNN:training()
    local parameters,gradParameters = g.featureCNN:getParameters()
    local xbatch, ybatch, masks = nil, nil, nil

    local echo = false

    local feval = function(x) 
        if ( x ~= parameters) then
            parameters:copy(x)
        end
        
        -- reset gradients
        gradParameters:zero()
        local xinput = g.fixedCNN:forward(xbatch)
        local f = g.featureCNN:forward(xinput)
     
        local totalLoss = {}
        local dfs = {}
        for i = 1, #f do
            
            if (i % 2) == 0 then
                print(masks[i][3]:sum()) 
            end

            local loss = g.lossLayers[i]:forward(f[i], ybatch[i])
            local df = g.lossLayers[i]:backward(f[i], ybatch[i])
                
            table.insert(totalLoss, loss)
            table.insert(dfs, df)
        end
       
        print(totalLoss)
        os.exit(0)

        g.featureCNN:backward(xinput, dfs)
        
        if echo then
            print(totalLoss)
        end

        return totalLoss, gradParameters
    end
    
    for i = 1, itnum do
        local batch = g.dataLoader:getBatch()
        xbatch = batch[1]:cuda()
        ybatch = batch[2]
        masks = batch[3]
        for j = 1, #ybatch do
            ybatch[j] = ybatch[j]:cuda()
            masks[j] = masks[j]:cuda()
        end
        
        if ( i % 100 ) == 0  then
            echo = true
        else
            echo = true
        end

        g.optim(feval, parameters, g.optimState) 
        collectgarbage();
        xlua.progress(i, itnum)
    end
end

local main = function()
    g.optim = optim.adam
    g.optimState = {
        learningRate = 0.0001
    }
    
    -- cuda 
    g.fixedCNN:cuda()
    g.featureCNN:cuda()
    for i = 1, #g.lossLayers do
        g.lossLayers[i]:cuda() 
    end
    
    for e = 1, 1 do
        doTrain(100)
    end
end 

main()

