require('torch')
require('image')
local u = require('./util')
local d = require('./data')
local m = require('./model')

-- Checking input paramters and load config
local cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-d', '', 'The target dataset folder')
cmd:option('-neck', 128, 'The middle hidden vector')
cmd:option('-total_iterator', 100000, "Total iterate number")
cmd:option('-batch_size', 16, "Batch number")
cmd:option('-seed', 1979, "Random seed")
local opt = cmd:parse(arg)
assert(opt.d ~= '')
local config = u.loadConfig(opt)
torch.manualSeed(opt.seed)
torch.setnumthreads(1)
torch.setdefaulttensortype('torch.FloatTensor')


local x = torch.rand(1, 3, config.inputWidth, config.inputHeight)
local netE = m.buildEncoder(opt, config)
netE:evaluate();

local y = netE:forward(x)
print(y:size())
