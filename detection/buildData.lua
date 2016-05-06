require 'image'
local flags = require './flags'

local buildSample = function(allDB, i) 
    local picInfo = allDB[i]

    local xmin, ymin, xmax, ymax = -1, -1, -1, -1
    
    for j = 1,#picInfo.boxes do
        if ( picInfo.boxes[j].xmin < xmin or xmin < 0) then
            xmin = picInfo.boxes[j].xmin
        end

        if ( picInfo.boxes[j].ymin < ymin or ymin < 0) then
            ymin = picInfo.boxes[j].ymin
        end
        
        if ( picInfo.boxes[j].xmax > xmax or xmax < 0) then
            xmax = picInfo.boxes[j].xmax
        end

        if ( picInfo.boxes[j].ymax > ymax or ymax < 0) then
            ymax = picInfo.boxes[j].ymax
        end
    end

    local bx, by, ex, ey = -1, -1, -1, -1
    if ( picInfo.width > picInfo.height ) then
        by = 1
        ey = flags.imageHeight
    else
        bx = 1
        ex = flags.imageWidth
    end
    if ( bx < 0) then
        local bxmin, bxmax = -1, -1
        
        bxmin = math.min( xmin, picInfo.width - flags.imageWidth + 1)
        bxmax = math.min( xmax, picInfo.width - flags.imageWidth + 1)

        bx = bxmin + math.floor( math.random()* (bxmax - bxmin) )  
        ex = bx + flags.imageWidth - 1
    else
        local bymin, bymax = -1, -1
        
        bymin = math.min( ymin, picInfo.height - flags.imageHeight + 1)
        bymax = math.min( ymax, picInfo.height - flags.imageHeight + 1)
        
        by = bymin + math.floor( math.random()* (bymax - bymin) )  
        ey = by + flags.imageHeight - 1
    end
    
    local img = image.loadJPG( flags.imagePath .. "/" .. picInfo.filename)
    img = image.crop(img, bx-1, by-1, ex, ey)

    -- build target vector
    local score = torch.Tensor(1600)
    for j = 1,#picInfo.boxes do
         
    end

end

function buildData(allDB, batch)
    local intputs = {}
    local targets = {}

    for i = 1, #batch do
        local input, target = buildSample(allDB, batch[i])
        if input ~= nil and target ~= nil then
            table.insert(inputs, input)
            table.insert(targets, target)
        end
    end

    local batchInput = torch.Tensor(#inputs, 3, flags.imageHeight, flags.imageWidth)
    local batchTarget = torch.Tensor(#targets, 1600)  
    for i = 1, #input do
        batchInput[i]:copy( inputs[i] )
    end
    for i = 1, #targets do
        batchTarget[i]:copy( targets[i] )
    end

end

local json = require "cjson"
local util = require "cjson.util"
local boxDB = json.decode(util.file_load(flags.allDB))
for i = 1,#boxDB do 
    buildSample(boxDB, i)
end

return buildData
