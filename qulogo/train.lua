require('torch')
require('image')
local u = require('./util')
local d = require('./data')
local m = require('./model')

-- Checking input paramters and load config
local cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-d', 'sohu', 'The target dataset folder')
cmd:option('-neck', 384, 'The middle hidden vector')
cmd:option('-total_iterator', 100000, "Total iterate number")
cmd:option('-batch_size', 16, "Batch number")
cmd:option('-seed', 1979, "Random seed")
local opt = cmd:parse(arg)
assert(opt.d ~= '')
local config = u.loadConfig(opt)
torch.manualSeed(opt.seed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')

-- Checking model input and output size
--[[
local x = torch.rand(5, 3, config.inputWidth, config.inputHeight)
local gen = m.buildGenerator(opt, config)
local enc = m.buildEncoder(opt, config)
local disc = m.buildDiscriminator(opt, config)
local z = enc:forward(x)
local y = gen:forward(z)
local v = disc:forward(y)
print(z:size(), y:size(), v:size())
--]]


