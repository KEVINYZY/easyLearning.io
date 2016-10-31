require('torch')
require('image')

local batchSize = 48
local typicalSize = 40      -- work with unblanced labeling

local targetWidth = 224
local targetHeight = 224

torch.setdefaulttensortype('torch.FloatTensor')
local dataProcessor = {}

local img2caffe = function(img)
    local mean_pixel = torch.Tensor({103.939, 116.779, 123.68})
    local perm = torch.LongTensor{3, 2, 1}
    img = img:index(1, perm):mul(256.0)
    mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
    img:add(-1, mean_pixel)
    return img
end

dataProcessor._init = function()
    local self = dataProcessor
    
    self.otherNumber = #self.otherSamples
    self.perm_o = torch.randperm(self.otherNumber)
    self.pos_o = 1
    
    self.typicalNumber = #self.typicalSamples 
    self.perm_t = torch.randperm(self.typicalNumber)
    self.pos_t = 1

    self.verifyNumber = #self.verifySamples
    self.perm_v = torch.randperm(self.verifyNumber)
    self.pos_v = 1
end

dataProcessor.doVerifySampling = function() 
    local self = dataProcessor
    local xBatch = torch.Tensor(batchSize, 3, targetHeight, targetWidth)
    local yBatch = {}
    for i=1,4 do
        table.insert(yBatch, torch.Tensor(batchSize))
    end

    for i=1,batchSize do
        local ii = self.perm_v[self.pos_v] 
        local info = self.verifySamples[ii]

        local fileName = info['name']
        local img = self._processImage(fileName, true)
        xBatch[i]:copy(img)

        yBatch[1][i] = info['pose']
        yBatch[2][i] = info['color']
        yBatch[3][i] = info['maker']
        yBatch[4][i] = info['type']

        self.pos_v = self.pos_v + 1
        if ( self.pos_v > self.verifyNumber) then
            self.pos_v = 1
        end
    end
 
    collectgarbage();
    return {xBatch, yBatch}  
end

dataProcessor.doSampling = function()
    local self = dataProcessor
    local xBatch = torch.Tensor(batchSize, 3, targetHeight, targetWidth)
    local yBatch = {}
    for i=1,4 do
        table.insert(yBatch, torch.Tensor(batchSize))
    end
    local fBatch = {}
    
    local i = 1 
    while true do
        if ( i  > batchSize ) then
            break;
        end

        local ii = self.perm_t[self.pos_t] 
        local info = self.typicalSamples[ii]
        
        if ( info['color'] == 1 ) then
            local fileName = info['name']
            table.insert(fBatch, fileName)
            local img = self._processImage(fileName)
            xBatch[i]:copy(img)

            yBatch[1][i] = info['pose']
            yBatch[2][i] = info['color']
            yBatch[3][i] = info['maker']
            yBatch[4][i] = info['type']

            i = i + 1
        end

        self.pos_t = self.pos_t + 1
        if ( self.pos_t > self.typicalNumber) then
            self.pos_t = 1
        end
    end

    collectgarbage();
    return {xBatch, yBatch, fBatch}
end

-- image random sampling 
dataProcessor._processImage = function(fileName, orignal)
    local img = image.loadJPG('./images/' .. fileName)
    
    local wid = img:size()[3]
    local hei = img:size()[2]

    -- scale and crop
    local sampleWidth = math.floor((1 - 0.15*math.random())*wid)
    local sampleHeight = math.floor((1 - 0.15*math.random())*hei)
    local sampleLeft = math.floor( (wid - sampleWidth) * math.random() )
    local sampleTop = math.floor( (hei - sampleHeight) * math.random() )
    
    local rotatedImg = image.rotate(img, (math.random()-0.5)*0.5)
    local cropedImg = image.crop(rotatedImg, sampleLeft, sampleTop, sampleLeft + sampleWidth, sampleTop + sampleHeight);
    
    local targetImg = image.scale(cropedImg, targetWidth, targetHeight)
    if ( math.random() > 0.5) then
        targetImg = image.hflip(targetImg)
    end
    targetImg = img2caffe(targetImg)
    
    return targetImg;
end

dataProcessor.otherSamples = infoDB[1]
dataProcessor.typicalSamples = infoDB[2]
dataProcessor.verifySamples = infoDB[3]

dataProcessor._init()

return dataProcessor
