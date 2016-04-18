require 'torch'
require 'image'
require 'nn'


local ConvLayer = function(in_size, out_size) 
    local conv = nn.SpatialConvolutionMM(in_size, out_size, 1, 1)
    conv.weight:normal(0, 1.0)

    return conv
end

local buildNet = function(input_size, output_size, net_number, layer_number) 
    local net = nn.Sequential()

    net:add( ConvLayer(input_size, net_number) )
    net:add( nn.Tanh() )
    for i = 1, layer_number - 1 do
        net:add( ConvLayer(net_number, net_number) )
        net:add( nn.Tanh() )
    end
    net:add( ConvLayer(net_number, output_size) )
    net:add( nn.Sigmoid() )

    return net
end

local buildXYRInput = function(width, height) 
    local xyr_in = torch.Tensor(3, height, width)

    for i = 1, width do
        xyr_in[{1, 1, i}] = (i - width/2) 
    end
    for i = 2, height do
        xyr_in[{1, i, {}}]:copy( xyr_in[{1, 1, {}}] )
    end

    for i = 1, height do
        xyr_in[{2, i, 1}] = (i - height/2) 
    end
    for i = 2, width do
        xyr_in[{2, {}, i}]:copy( xyr_in[{2, {}, 1}] )
    end
   
    xyr_in[3] = torch.cmul(xyr_in[1], xyr_in[1] ) + torch.cmul(xyr_in[2], xyr_in[2])
    xyr_in[3]:sqrt()

    return xyr_in
end
local net = buildNet(3, 3, 32, 5)
local xyr = buildXYRInput(640, 640)
xyr = xyr / 500.0
local out = net:forward(xyr)
image.savePNG('./output.png',out)

