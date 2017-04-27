import torch
import torch.nn as nn
from torch.autograd import Variable as V
from torch.autograd import Function as F

import pttoc

fc1 = nn.Linear(10,20)
fc1.weight.data.normal_(0.0,1.0)
fc1.bias.data.normal_(0.0,1.0)

fc2 = nn.Linear(20,2)
fc2.weight.data.normal_(0.0,1.0)
fc2.bias.data.normal_(0.0,1.0)

data = V(torch.rand(10, 10))

x1 = torch.nn.functional.relu(fc1(data))
model = lambda x: torch.nn.functional.log_softmax(fc2(x1))

x2 = fc2(x1)
y = model(data)

g = pttoc.buildGraph([data], [y, x2])


