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
    img = img - 0.5
    img = img * 256
    
    -- build target value
    local targets = {}
    for i=1,flags.grid * flags.grid do
        targets[i] = 1
    end
    
    local boxPos = torch.Tensor( flags.grid * flags.grid * 4) 
    targets[flags.grid * flags.grid + 1] = boxPos

    for y = 1, flags.grid do
        for x = 1, flags.grid do
            local offset = ((y-1)*flags.grid + x - 1) * 4
            
            boxPos[offset + 1] = (x - 1) / flags.grid
            boxPos[offset + 2] = (y - 1) / flags.grid
            boxPos[offset + 3] = x / flags.grid
            boxPos[offset + 4] = y / flags.grid
        end
    end
    
    local objs = 0
    for i = 1,#picInfo.boxes do
        xmin = math.max(picInfo.boxes[i].xmin, bx) - bx
        xmax = math.min(picInfo.boxes[i].xmax, ex) - bx 
        
        ymin = math.max(picInfo.boxes[i].ymin, by) - by
        ymax = math.min(picInfo.boxes[i].ymax, ey) - by
    
        local cx = (xmin + xmax)/2 
        local cy = (ymin + ymax)/2 
        cx = math.floor( cx * flags.grid / flags.imageWidth) 
        cy = math.floor( cy * flags.grid / flags.imageHeight)

        if ( cx >= 0 and cx < flags.grid and cy >= 0 and cy < flags.grid) then
            objs = objs + 1 
            
            local offset = cx + cy * flags.grid + 1
            local class = flags.classmap[ picInfo.boxes[i].name ]
            targets[offset] = class + 1
            
            offset = (offset - 1) * 4

            boxPos[offset + 1] = xmin / flags.imageWidth
            boxPos[offset + 2] = ymin / flags.imageHeight
            boxPos[offset + 3] = xmax / flags.imageWidth
            boxPos[offset + 4] = ymax / flags.imageHeight
        end
    end
    
    if objs == 0 then
        return nil
    end
    

    return img, targets
end

function buildData(allDB, batch)
    local inputs = {}
    local targets = {}

    for i = 1, #batch do
        local input, target = buildSample(allDB, batch[i])
        if input ~= nil and target ~= nil then
            table.insert(inputs, input)
            table.insert(targets, target)
        end
    end

    local batchInput = torch.Tensor(#inputs, 3, flags.imageHeight, flags.imageWidth)
    for i = 1, #input do
        batchInput[i]:copy( inputs[i] )
    end
   
    local batchTargets = {}
    for i=1, flags.grid * flags.grid do
        batchTargets[i] = torch.Tensor(#inputs)
        for j = 1, #inputs do
            batchTargets[i][j] = targets[j][i] 
        end
    end
    
    local posTarget = torch.Tensor(#inputs, flags.grid * flags.grid * 4)
    batchTargets[flags.grid * flags.grid + 1] = posTarget
    for i = 1, #inputs do
        posTarget[i]:copy( targets[i][ flags.grid * flags.grid + 1 ] ) 
    end
    
    if flags._cuda_ then
        for i = 1, #batchTargets do
            batchTargets[i] = batchTargets[i]:cuda()
        end
        batchInput = batchInput:cuda()
    end

    return batchInput, batchTargets
end

--[[
local json = require "cjson"
local util = require "cjson.util"
local boxDB = json.decode(util.file_load(flags.allDB))
buildData(boxDB, {1,2,3,4})
--]]

return buildData
