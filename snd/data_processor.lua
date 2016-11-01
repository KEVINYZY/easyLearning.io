require('torch')
require('image')

local boxSampling = require('boxsampling')

local batchSize = 16
local allShapes = { {256, 256} ,
                    {224, 288} ,
                    {288, 224} }

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

dataProcessor._init = function(modelInfo)
    local self = dataProcessor
    self.modelInfo = modelInfo
    
    self.trainNumber = #self.trainSamples 
    self.trainPerm = torch.randperm(self.trainNumber)
    self.trainPos = 1

    self.verifyNumber = #self.verifySamples
    self.verifyPerm = torch.randperm(self.verifyNumber)
    self.verifyPos = 1
end

dataProcessor.doSampling = function()
    local self = dataProcessor
   
    local _ = math.floor(math.random() * 100) % 3 + 1
    local targetWidth = allShapes[_][1]
    local targetHeight = allShapes[_][2]

    local i = 1
    while ( i <= 32 ) do
        local ii = self.trainPerm[self.trainPos] 
        local info = self.trainSamples[ii]

        local targetImg, labels = self._processImage(info, targetWidth, targetHeight)
                
        if ( #labels > 0) then
            i = i + 1
            local predBoxes = boxSampling(self.modelInfo, info.image.width, info.image.height, labels)
            
        end

        self.trainPos = self.trainPos + 1
        if ( self.trainPos > self.trainNumber) then
            self.trainPos = 1
        end
    end

    collectgarbage();
end

dataProcessor.doVerifySampling = function()
    local self = dataProcessor
    
    
end

-- image random sampling 
dataProcessor._processImage = function(info, targetWidth, targetHeight)
    local img = image.loadJPG('./data/' .. info['image']['file'])
    local wid = img:size()[3]
    local hei = img:size()[2]
   
    -- scale and crop
    local scale = targetWidth / wid
    local cutWid, cutHei = targetWidth , math.floor(hei * scale)
    local offsetx, offsety = 0, math.floor(math.random() * (cutHei - targetHeight) )
    if ( cutHei < targetHeight) then
        scale = targetHeight / hei
        cutWid, cutHei = math.floor(wid * scale), targetHeight
        offsety, offsetx = 0, math.floor(math.random() * (cutWid - targetWidth) )
    end
    local scaledImg = image.scale(img, cutWid, cutHei)
    local targetImg = image.crop(scaledImg, offsetx, offsety, offsetx + targetWidth, offsety + targetHeight) 

    local labels = {}
    local anns = info["annotation"]
    for i = 1, #anns do
        local bbox = {}
        bbox.class = anns[i]['category_id']
   
        bbox.xmin = math.floor(anns[i]['bbox'][1]*scale) - offsetx
        bbox.ymin = math.floor(anns[i]['bbox'][2]*scale) - offsety
        bbox.xmax = math.floor(anns[i]['bbox'][3]*scale) + bbox.xmin 
        bbox.ymax = math.floor(anns[i]['bbox'][4]*scale) + bbox.ymin
        
        if ( bbox.xmax > 0 and bbox.ymax > 0 and bbox.xmin < targetWidth and bbox.ymin < targetHeight ) then
            if ( bbox.xmin < 0) then
                bbox.xmin = 0
            end
            if ( bbox.ymin < 0) then
                bbox.ymin = 0
            end
            if ( bbox.xmax >= targetWidth ) then
                bbox.xmax = targetWidth
            end
            if ( bbox.ymax >= targetHeight ) then
                bbox.ymax = targetHeight
            end
            
            table.insert(labels, bbox)
        end
    end

    if ( math.random() > 0.5) then
        targetImg = image.hflip(targetImg)
        for i = 1, #labels do
            local temp = targetWidth - labels[i].xmin
            labels[i].xmin = targetWidth - labels[i].xmax
            labels[i].xmax = temp
        end
    end

    --[[
    for i = 1, #labels do
        local bbox = labels[i]
        targetImg = image.drawRect(targetImg, bbox.xmin, bbox.ymin, bbox.xmax, bbox.ymax);  
    end
    local randFile = './images/' .. math.random() .. '.jpg'
    image.save(randFile, targetImg)
    --]]
    
    targetImg = img2caffe(targetImg)
    return targetImg, labels
end

dataProcessor.trainSamples = infoDB[1]
dataProcessor.verifySamples = infoDB[2]

return dataProcessor

