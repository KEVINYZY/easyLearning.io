require('torch')
require('image')

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
    
end

dataProcessor.doSampling = function()
    local self = dataProcessor
    
    return nil
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

dataProcessor._init()

return dataProcessor
