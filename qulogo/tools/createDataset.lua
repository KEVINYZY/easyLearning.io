require('torch')
require('lfs')
require('image')
local json = require('cjson')

local cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-i', '', 'The target media file')
cmd:option('-d', '', 'The target media file')
cmd:option('-x', -1, "Logo's x")
cmd:option('-y', -1, "Logo's y")
cmd:option('-width', -1, "Logo's width")
cmd:option('-height', -1, "Logo's height")
local opt = cmd:parse(arg)

assert(opt.i ~= '', 'Miss input file')
assert(opt.d ~= '', 'Miss output floder')
assert(opt.x >= 0,  'Logo x < 0')
assert(opt.y >= 0,  'Logo y < 0')
assert(opt.width > 0,   'Logo width < 0')
assert(opt.height > 0,   'Logo height < 0')

print("Create dataset floder...")
os.execute('mkdir ' .. opt.d)
os.execute('rm -f ' .. opt.d .. '/*.jpg')

print("Extracing images from video file...")
local cmd = 'ffmpeg -y -i ' .. opt.i .. ' ' .. opt.d .. '/%d.jpg'
os.execute(cmd)

-- Counting total images
cmd = 'find ' .. opt.d .. ' -name "*.jpg" > /tmp/img.list'
os.execute(cmd)

local fh = io.open('/tmp/img.list')
local totalNumber = 0
while true do
     line = fh:read()
     if line == nil then break end
     totalNumber = totalNumber + 1
end
local tempImage = image.loadJPG(opt.d .. '/1.jpg')
local size = tempImage:size();

config = {}
config.totalNumber = totalNumber
config.imageWidth = size[2]
config.imageHeigth = size[3]
config.logoX = opt.x
config.logoY = opt.y
config.logoWidth = opt.width
config.logoHeight = opt.height

fh = io.open(opt.d .. '/config.json', 'w')
fh:write( json.encode(config) );
fh:close();


