require('torch')
require('cunn')
require('image')

local processImage = function(fileName, targetWidth, targetHeight)
    local img2caffe = function(img)
        local mean_pixel = torch.Tensor({103.939, 116.779, 123.68})
        local perm = torch.LongTensor{3, 2, 1}
        img = img:index(1, perm):mul(256.0)
        mean_pixel = mean_pixel:view(3, 1, 1):expandAs(img)
        img:add(-1, mean_pixel)
        return img
    end

    local img = image.loadJPG(fileName)
    local wid = img:size()[3]
    local hei = img:size()[2]
   
    local scale = 0
    if ( wid > hei ) then
        scale = 224 / hei
    else
        scale = 224 / wid
    end
    local newWid = scale * wid
    local newHei = scale * hei
    
    if ( newWid < 224) then 
        newWid = 224
    end
    if ( newHei < 224 ) then
        newHei = 224
    end
    
    newWid = newWid - (newWid % 32)
    newHei = newHei - (newHei % 32)
    
    local scaledImg = image.scale(img, newWid, newHei)
    local targetImg = img2caffe(scaledImg)

    return targetImg, scaledImg
end 


-- init
torch.setdefaulttensortype('torch.FloatTensor')
cutorch.setDevice(1)

local _ = require('./model.lua')
local modelInfo = _.info
local boxSampling = require('boxsampling')

local xinput, origin = processImage( arg[1]) 
_ = xinput:size()

local targetWidth = _[3]
local targetHeight = _[2]
local predBoxes = boxSampling( modelInfo, targetWidth, targetHeight)  

local snd = torch.load('models/fullModel.t7')
snd:cuda()
local yout = snd:forward(xinput:cuda())

local _ = modelInfo.getSize(targetWidth, targetHeight)
local lastWidth = _[1]
local lastHeight = _[2]

local bestInfo = {}
for i = 1, #modelInfo.boxes do
    local wid = lastWidth - (modelInfo.boxes[i][1] - 1)
    local hei = lastHeight - (modelInfo.boxes[i][2] - 1)
   
    local conf = yout[(i-1)*2 + 1]:float()

    for h = 1,hei do
        for w = 1,wid do
            local smf = conf[{{}, h, w}]:reshape(modelInfo.classNumber)
            local v,_ = smf:max(1)
            if ( _[1] ~= modelInfo.classNumber ) then
                if ( bestInfo.v == nil or v[1] > bestInfo.v ) then
                    bestInfo.v = v[1]
                    bestInfo.h = h
                    bestInfo.w = w
                    bestInfo.i = i
                    bestInfo.c = _[1]
                end
            end
        end
    end
end

local ii = "_" .. bestInfo.i .. "_" .. bestInfo.h .. "_" .. bestInfo.w
local box = predBoxes[ii]
local h = bestInfo.h
local w = bestInfo.w

local loc = yout[bestInfo.i*2]:float()
local xmin = loc[1][h][w]*16 + box.xmin
local ymin = loc[2][h][w]*16 + box.ymin
local xmax = loc[3][h][w]*16 + box.xmax
local ymax = loc[4][h][w]*16 + box.ymax

print( bestInfo )

local img = image.drawRect(origin, xmin, ymin, xmax, ymax)
image.save('/tmp/1.jpg', img);

