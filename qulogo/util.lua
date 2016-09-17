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
    longSide = math.floor((longSide + 7) / 8) * 8.0

    local x = config.logoX - longSide*1.5
    local y = config.logoY - longSide*1.5
    if ( x < 0 ) then x = 0 end
    if ( y < 0 ) then y = 0 end
    if ( x + longSide*3 > config.imageWidth ) then
        x = config.imageWidth - longSide*4
    end
    if ( y + longSide*3 > config.imageHeight ) then
        y = config.imageHeight - longSide*4
    end
    config.inputWidth = longSide*4
    config.inputHeight = longSide*4

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
