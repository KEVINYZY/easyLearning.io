from __future__ import print_function
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from torch.utils.data import DataLoader

from dataset import DatasetFromFolder
from model import GuestureNet

# Training settings
parser = argparse.ArgumentParser(description='PyTorch Guesture example')
parser.add_argument('--batchSize', type=int, default=12, help='training batch size')
parser.add_argument('--nEpochs', type=int, default=10, help='number of epochs to train for')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--lr', type=float, default=0.001, metavar='LR', help='learning rate (default: 0.01)')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
opt = parser.parse_args()

# init & setup
torch.manual_seed(opt.seed)
dataLoader = DataLoader(dataset=DatasetFromFolder('./guesture'), num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True)

model = GuestureNet()
optimizer = optim.Adam(model.parameters(), lr=opt.lr)

def train(epoch):
    model.train()
    for iteration, batch in enumerate(dataLoader):
        data = Variable(batch[0])
        target = Variable(batch[1])
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        print("loss value = {}".format(loss[0]))

def checkpoint(epoch):
    model_out_path = "model_epoch_{}.pth".format(epoch)
    torch.save(model, model_out_path)
    print("Checkpoint saved to {}".format(model_out_path))

for epoch in range(1, opt.nEpochs + 1):
    train(epoch)
    checkpoint(epoch)
