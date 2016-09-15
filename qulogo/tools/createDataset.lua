require('torch')

local cmd = torch.CmdLine()

cmd:text('Options:')
cmd:option('-i', '', 'The target media file')
cmd:option('-d', '', 'The target media file')
cmd:option('-x', -1, "Logo's x")
cmd:option('-y', -1, "Logo's y")
cmd:option('-w', -1, "Logo's width")
cmd:option('-h', -1, "Logo's height")

local opt = cmd:parse(arg)

assert(opt.i ~= '', 'Miss input file')
assert(opt.d ~= '', 'Miss output floder')
assert(opt.x >= 0,  'Logo x < 0')
assert(opt.y >= 0,  'Logo y < 0')
assert(opt.w < 0,   'Logo width < 0')
assert(opt.h < 0,   'Logo height < 0')


