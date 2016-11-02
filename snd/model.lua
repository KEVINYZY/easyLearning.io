require('nn')
require('cunn')
require('image')

torch.setdefaulttensortype('torch.FloatTensor')

local classNumber = 21

local fixedCNN = torch.load('fixedCNN.t7');
fixedCNN:evaluate()

-- input is 56x56
local featureCNN = nn.Sequential()

featureCNN:add( nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1) )
featureCNN:add( nn.ReLU(true) )
featureCNN:add( nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1) )
featureCNN:add( nn.ReLU(true) )
featureCNN:add( nn.SpatialConvolution(512, 512, 3, 3, 1, 1, 1, 1) )
featureCNN:add( nn.ReLU(true) )
featureCNN:add( nn.SpatialMaxPooling(2, 2, 2, 2, 0, 0):ceil() )

-- output is 7x7

featureCNN:add(nn.SpatialConvolution(512, 1024, 1, 1, 1, 1, 0, 0))
featureCNN:add(nn.LeakyReLU(0.1))

local mbox = nn.ConcatTable()
local allBoxes = { {1,1}, {2,2}, {3,3}, {4,4}, {5,5}, {6,6}, {7,7}, 
                   {1,2}, {2,1}, {2,4}, {4,2}, {3,6}, {6,3},
                   {1,3}, {3,1}, {2,6}, {6,2},
                   {2,3}, {3,2}, {4,6}, {6,4},
                   {2,5}, {5,2}, {3,4}, {4,3} }

for i = 1, #allBoxes do
    local boxConf = nn.Sequential()
    boxConf:add(nn.SpatialConvolution(1024, classNumber, allBoxes[i][1], allBoxes[i][2], 1, 1, 0, 0))
    boxConf:add(nn.SpatialLogSoftMax())
    mbox:add(boxConf)
 
    local boxLoc = nn.Sequential()
    boxLoc:add(nn.SpatialConvolution(1024,  4, allBoxes[i][1], allBoxes[i][2], 1, 1, 0, 0))
    mbox:add(boxLoc)
end
featureCNN:add(mbox)

local getSize = function(imageWidth, imageHeight) 
    local targetWidth = math.floor(imageWidth/32) 
    local targetHeight = math.floor(imageHeight/32)
    
    return {targetWidth, targetHeight};
end

--[[
fixedCNN:cuda()
featureCNN:cuda()
local x = torch.rand(4, 3,256,256):cuda()
local y = featureCNN:forward( fixedCNN:forward(x) )
print(y)
--]]

local model = {}
model.fixedCNN = fixedCNN
model.featureCNN = featureCNN

local info = {}
info.classNumber = classNumber
info.boxes = allBoxes
info.getSize = getSize
model.info = info

return model
