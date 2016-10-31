require('torch')

local jaccardOverlap = function(b1, b2)
    local xmin = math.min(b1.xmin, b2.xmin)
    local ymin = math.min(b1.ymin, b2.ymin)
    local xmax = math.max(b1.xmax, b2.xmax)
    local ymax = math.max(b1.ymax, b2.ymax)

    -- no overlap
    if ( (xmax - xmin) > ( b1.xmax - b1.xmin + b2.xmax - b2.xmin) or
        (ymax - ymin) > ( b1.ymax - b1.ymin + b2.ymax - b2.ymin) ) then
        return 0
    end
    
    local w1 = b1.xmax - b1.xmin
    local h1 = b1.ymax - b1.ymin
    local w2 = b2.xmax - b2.xmin
    local h2 = b2.ymax - b2.ymin
    
    local andw = w1 + w2 - (xmax - xmin)
    local andh = h1 + h2 - (ymax - ymin)

    local andArea = and2 * andh
    local orArea = w1 * h1 + w2*h2 - aArea
    
    local overlap = andArea / orArea
end
    
local buildTarget = function(model, imageWidth, imageHeight, labels) 
    local _ = model.getSize(imageWidth, imageHeight)
    local targetWidth = _[1]
    local targetHeight = _[2]
    
    local cellWidth = imageWidth / targetWidth
    local cellHeight = imageHeight / targetHeight
    
    -- build all pred boxes
    local predBoxes = {}
    local target = {}
    for i = 1, #model.boxes do
        local wid = targetWidth - (model.boxes[i][1] - 1)
        local hei = targetHeight - (model.boxes[i][2] - 1)
        
        local confTarget = torch.Tensor(model.classNumber, wid, hei)
        local locTarget = torch.Tensor(4, wid, hei)

        table.insert(target, confTarget)
        table.insert(target, locTarget)

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
    
    -- find best match between labels (groud truth) and predBoxes
    local matchMap = torch.zeros(#predBoxes, #labels) - 1
    for i = 1, #predBoxes do
        for j = 1, #labels do
            local overlap = jaccardOverlap(predBoxes[i], labels[j]) 
            if ( overlap > 0.0001 ) then
                matchMap[i][j] = overlap
            end
        end
    end
    
    matchMap = matchMap:reisze(#predBoxes * #label)
    
    local postiveBoxes = {}

    -- best match from bidirection
    for i = 1, #labels do
        local tempMap = matchMap:reshape(#predBoxes * #label)
        local score, _ = tempMap:max(1);
        local p = math.floor( (_[1] - 1) / #label) + 1
        local l = _[1] - (y - 1) * #label 
        matchMap[{{}, l}] = -1;
        matchMap[{p, {}}] = -1;
        table.insert( postiveBoxes, {p, l, i})
    end
    -- select rest best match > threshold as postive
    for i = 1, #predBoxes do
        local score, _ = matchMap[i]:max(1)
        if ( score > 0.5 ) then
            table.insert(postiveBoxes, {i, _[1], _[1]})
            matchMap[i] = -1
        end
    end
   
    local negativeBoxes = {}
    
    -- do negative sampling
    local tempMap = matchMap:reshape(#predBoxes * #label)
    local _, ii = tempMap:sort(true)
    for i = 1, #postiveBoxes * 3 do
        if ( _[i] > 0.0001 ) then
            local p = math.floor( (_[1] - 1) / #label) + 1
            table.insert(negativeBoxes, p); 
        else
            break;
        end
    end

    return postiveBoxes, negativeBoxes
end


