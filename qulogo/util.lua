require('torch')
require('image')
local json = require('cjson')

local loadConfig = function(opt) 
    -- Loading dataset's config file
    local fh = io.open(opt.d .. '/config.json')
    local configString = fh:read("*all")
    local config = json.decode(configString)

    -- Computing logo black mask (zero index)
    local longSide = math.max(config.logoWidth, config.logoHeight)
    --if ( longSide%2 ~= 0 ) then longSide = longSide + 1 end
    if ( longSide <= 48) then 
        longSide = 48 
    elseif ( longSide <= 64) then
        longSide = 64
    elseif ( longSide <= 80) then
        longSide = 80
    else
        longSide = -1
    end
    assert(longSide > 0)
    
    local x = config.logoX - longSide
    local y = config.logoY - longSide
    if ( x < 0 ) then x = 0 end
    if ( y < 0 ) then y = 0 end
    if ( x + longSide*3 > config.imageWidth ) then
        x = config.imageWidth - longSide*3
    end
    if ( y + longSide*3 > config.imageHeight ) then
        y = config.imageHeight - longSide*3
    end
    config.inputWidth = longSide*3
    config.inputHeight = longSide*3

    -- mask value is for tensor, so it is one index
    config.maskLeft = config.logoX - x + 1
    config.maskTop = config.logoY - y + 1
    config.maskWidth = longSide
    config.maskHeight = longSide
    assert(config.maskTop >= 1)
    assert(config.maskLeft >= 1)
     

    return config
end

local util = {}
util.loadConfig = loadConfig

return util
