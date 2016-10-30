require('torch')


local getSize = function(imageWidth, imageHeight) 
    local targetWidth = math.floor(imageWidth/32) 
    local targetHeight = math.floor(imageHeight/32)
    
    return {targetWidth, targetHeight};
end


local buildTarget = function(model, imageWidth, imageHeight, label) 
    local _ = model.getSize(imageWidth, imageHeight)
    local targetWidth = _[1]
    local targetHeight = _[2]
    
    local cellWidth = imageWidth / targetWidth
    local cellHeight = imageHeight / targetHeight
    
    -- build all pred boxes
    local predBoxes = {}
    for i = 1, #model.boxes do
        local wid = targetWidth - (model.boxes[i][1] - 1)
        local hei = targetHeight - (model.boxes[i][2] - 2)
 
        for w = 1, wid do
            for h = 1, hei do
                local box = {}
                box.bindex = i
                box.xmin = (w-1) * cellWidth + 1
                box.ymin = (h-1) * cellHeight + 1
                box.xmax = (w + model.boxes[i][1] - 1) * cellWidth
                box.ymax = (h + model.boxes[i][2] - 1) * cellHeight
                
                table.insert(predBoxes, box)
            end
        end
    end
    
    -- find best match to label 

end

local allBoxes = { {2,2}, {3,3}, {4,4}, {5,5}, {6,6}, {7,7}, 
                   {2,4}, {4,2}, 
                   {3,6}, {6,3} }

local model = {}
model.getSize = getSize
model.boxes = allBoxes


buildTarget(model, 320, 320)

