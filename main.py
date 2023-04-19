from __future__ import print_function
import argparse

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR

from utils.config_utils import read_args, load_config, Dict2Object


# define the torch , inherit sth from the module ?
# 4 layers
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()  # similar to constructor in java
        self.conv1 = nn.Conv2d(1, 32, 3, 1)
        self.conv2 = nn.Conv2d(32, 64, 3, 1)
        self.dropout1 = nn.Dropout(0.25)
        self.dropout2 = nn.Dropout(0.5)  # drop out : avoid over fitting
        self.fc1 = nn.Linear(9216, 128)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = F.relu(x)  # activation layer
        x = self.conv2(x)
        x = F.relu(x)
        x = F.max_pool2d(x, 2)
        x = self.dropout1(x)
        x = torch.flatten(x, 1)
        x = self.fc1(x)
        x = F.relu(x)
        x = self.dropout2(x)
        x = self.fc2(x)
        output = F.log_softmax(x, dim=1)  # 最大可能性的那个数字
        return output


# 每一个epoch都需要所有data，所以我们divide data into batches
# use training data to update the model parameters
def train(args, model, device, train_loader, optimizer, epoch):
    """
    train the model and return the training accuracy
    :param args: input arguments
    :param model: neural network model
    :param device: the device where model stored
    :param train_loader: data loader
    :param optimizer: optimizer
    :param epoch: current epoch
    :return:
    """
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
    '''TODO: Fill your code'''
    if batch_idx % log_interval == 0:

    training_acc, training_loss = None, None  # replace this line:改成我得到的值
    return training_acc, training_loss


# data-loader= test loader
def test(model, device, test_loader):
    """
    test the model and return the testing accuracy
    :param model: neural network model
    :param device: the device where model stored
    :param test_loader: data loader
    :return:
    """
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            '''Fill your code'''
            data , target = data.to(device), target.to(device)
            output= model(data.to(device))
            test_loss += F.nll_loss(output, target.to(device), reduction='sum').item()
            pred = output.argmax(dim=1, keepDim=True)
            correct += pred.eq(target.view_as(pred)).sum().item()
            # pass
            # a variable is created and do not necessarily need to assign value
    test_loss/=len(test_loader.dataset)  # len: return the number of elements in a container
    testing_acc, testing_loss = None, None  # replace this line：比较
    return testing_acc, testing_loss


# plot function should generate line charts based on:
# 1.the records training loss
# 2.testing loss
# 3.testing accuracy
# The plot should include the tle and labels
def plot(epoches, performance):
    """
    plot the model performance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    """Fill your code"""
    # 横轴epoch 纵轴performance, 把之前算出来的数据用数组存起来然后画图
    xpoints= np.array([])
    ypoints= np.array([])
    plt.plot(xpoints, ypoints)
    plt.show()
    pass


def run(config):
    use_cuda = not config.no_cuda and torch.cuda.is_available()
    use_mps = not config.no_mps and torch.backends.mps.is_available()

    torch.manual_seed(config.seed)

    if use_cuda:
        device = torch.device("cuda")
    elif use_mps:
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    train_kwargs = {'batch_size': config.batch_size, 'shuffle': True}
    test_kwargs = {'batch_size': config.test_batch_size, 'shuffle': True}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True, }
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    # download data
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./data', train=True, download=True, transform=transform)
    dataset2 = datasets.MNIST('./data', train=False, transform=transform)

    """add random seed to the DataLoader, pls modify this function"""
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr) # try to find the best output

    """record the performance"""
    # 这几行都是老师的template
    epoches = []
    training_accuracies = []
    training_loss = []
    testing_accuracies = []
    testing_loss = []

    scheduler = StepLR(optimizer, step_size=1, gamma=config.gamma)
    for epoch in range(1, config.epochs + 1):
        train_acc, train_loss = train(config, model, device, train_loader, optimizer, epoch)
        """record training info, Fill your code"""
        training_accuracies[epoch]=train_acc, training_loss[epoch]=train_loss
        test_acc, test_loss = test(model, device, test_loader)
        """record testing info, Fill your code"""
        testing_accuracies[epoch]=test_acc, testing_loss[epoch]=test_loss
        scheduler.step()
        """update the records, Fill your code"""
        #  todo：不会update，怎么update啊

    """plotting training performance with the records"""
    plot(epoches, training_loss)

    """plotting testing performance with the records"""
    plot(epoches, testing_accuracies)
    plot(epoches, testing_loss)

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")


def plot_mean():
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    # 需要自己定义record data的函数
    """fill your code"""


if __name__ == '__main__':
    arg = read_args()

    """toad training settings"""
    config = load_config(arg)

    """train model and record results"""
    run(config)

    """plot the mean results"""
    plot_mean()
