require('torch')
require('lfs')
require('image')
local json = require('cjson')

local cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-d', '', 'The target dataset folder')
cmd:option('-o', '', 'The output images folder')
cmd:option('-skipf',  0, "Skip frames per batch")
cmd:option('-ep', 5, "Training 
cmd:option('-y', -1, "Logo's y")
cmd:option('-width', -1, "Logo's width")
cmd:option('-height', -1, "Logo's height")
local opt = cmd:parse(arg)

