require('torch')
require('image')
local u = require('./util')
local d = require('./data')

-- Checking input paramters and load config
local cmd = torch.CmdLine()
cmd:text('Options:')
cmd:option('-d', '', 'The target dataset folder')
cmd:option('-total_iterator', 100000, "Total iterate number")
cmd:option('-batch_size', 16, "Batch number")
local opt = cmd:parse(arg)
assert(opt.d ~= '')
local config = u.loadConfig(opt)


