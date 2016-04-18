require 'nn'

function build_model()
    
    local cnn = nn.Sequential()
    cnn:add(nn.SpatialConvolution(1, 8, 3, 3, 1, 1, 1))    
    cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(2, 2))
    cnn:add(nn.SpatialConvolution(8, 16, 3, 3, 1, 1, 1))    
    cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(2, 2))
    cnn:add(nn.SpatialConvolution(16, 16, 3, 3, 1, 1, 1))
    cnn:add(nn.ReLU())
    cnn:add(nn.SpatialMaxPooling(2, 2))
    cnn:add(nn.SpatialConvolution(16, 32, 3, 3, 1, 1, 1))
    cnn:add(nn.ReLU())
    
    cnn:add(nn.Reshape(32*4*4)) 
    cnn:add(nn.Linear(32*4*4, 512))
    cnn:add(nn.ReLU())
    cnn:add(nn.Linear(512, 10))
    --cnn:add(nn.Sigmoid())
    cnn:add(nn.Linear(10, 2))

    --The siamese model
    --clone the encoder and share the weight, bias. Must also share the gradWeight and gradBias
    siamese_encoder = nn.ParallelTable()
    siamese_encoder:add(cnn)
    siamese_encoder:add(cnn:clone('weight','bias', 'gradWeight','gradBias')) 

    --The siamese model (inputs will be Tensors of shape (2, channel, height, width))
    model = nn.Sequential()
    model:add(nn.SplitTable(2)) -- split input tensor along the rows (1st dimension) to table for input to ParallelTable
    model:add(siamese_encoder)
    model:add(nn.PairwiseDistance(2)) --L2 pariwise distance
    
    return model
end


return build_model


