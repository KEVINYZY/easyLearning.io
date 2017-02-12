import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

class GuestureNet(nn.Module):
    def __init__(self):
        super(GuestureNet, self).__init__()

        self.conv1_0 = nn.Conv2d(3, 32, (3, 3), stride=(1, 1), padding=(1,1))
        self.conv1_1 = nn.Conv2d(32, 32, (3, 3), stride=(1, 1), padding=(1,1))

        self.conv2_0 = nn.Conv2d(32, 64, (3, 3), stride=(1, 1), padding=(1,1))
        self.conv2_1 = nn.Conv2d(64, 128, (3, 3), stride=(1, 1), padding=(1,1))

        self.conv3_0 = nn.Conv2d(128, 128, (3, 3), stride=(1, 1), padding=(1,1))

        self.fc1 = nn.Linear(8192, 1024)
        self.fc2 = nn.Linear(1024, 6)

    def forward(self, x):
        x = F.relu(self.conv1_0(x))
        x = F.relu(self.conv1_1(x))
        x = F.max_pool2d(x, 2)              # output is 32x32

        x = F.relu(self.conv2_0(x))
        x = F.relu(self.conv2_1(x))
        x = F.max_pool2d(x, 2)              # output is 16x16

        x = F.relu(self.conv3_0(x))
        x = F.max_pool2d(x, 2)              # output is 8x8
        x = x.view(-1, 8192)                # size = batch x 8192
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.log_softmax(x)
        return x

if __name__ == '__main__':
    model = GuestureNet()
    x = torch.autograd.Variable( torch.rand(1, 3, 64, 64) )
    y = model(x)
    print(y)
