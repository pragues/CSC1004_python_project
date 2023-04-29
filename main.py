from __future__ import print_function

# 上面一行是自己加的
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
import os
import multiprocessing
import argparse

from config_utils import read_args, load_config, Dict2Object


class args_input:
    config_file = ''


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
    :param epoch: current epoch, 对每一个epoch进行分别的train
    :return:
    """
    sum_loss = 0
    sum_acc = 0
    model.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data)
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        sum_loss += loss.item()

    '''TODO: Fill your code'''
    count_train = 0
    for i in range(output.shape[0]):
        if torch.argmax(output[i], -1).item() == target[i].item():
            count_train += 1
    sum_acc += count_train / train_loader.batch_size
    print("loss: " + str(loss.item()))
    print("accuracy: " + str(sum_acc / len(train_loader.dataset)))
    with open("train_" + str(args.seed) + ".txt", "a") as f:
        f.write("loss: " + str(loss.item()) + " accuracy: " + str(sum_acc / len(train_loader.dataset)) + "\n")

    training_acc = sum_acc / len(train_loader.dataset)
    training_loss = sum_loss / len(train_loader.dataset) # replace this line:改成我得到的值
    return training_acc, training_loss


# data-loader= test loader
def test(args, model, device, test_loader):
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
            count = 0
            data, target = data.to(device), target.to(device)
            output = model(data)
            loss = F.nll_loss(output, target)
            test_loss += loss.item()
            for i in range(output.shape[0]):
                if torch.argmax(output[i], -1).item() == target[i].item():
                    count += 1
            correct += count / test_loader.batch_size
            print("loss: " + str(loss.item()))
            print("accuracy: " + str(correct / len(test_loader.dataset)))
            with open("test_" + str(args.seed) + ".txt", "a") as f:
                f.write("loss: " + str(loss.item()) + "accuracy: " + str(correct / len(test_loader.dataset)) + "\n")
            # pass
            # a variable is created and do not necessarily need to assign value
    # len: return the number of elements in a container
    testing_acc = correct / len(test_loader.dataset)
    testing_loss = test_loss / len(test_loader.dataset) # replace this line：比较
    return testing_acc, testing_loss


# plot function should generate line charts based on:
# 1.the records training loss
# 2.testing loss
# 3.testing accuracy
# The plot should include the tle and labels
def plot(epoches, performance, title):
    """
    plot the model performance
    :param epoches: recorded epoches
    :param performance: recorded performance
    :return:
    """
    """Fill your code"""
    # 横轴epoch 纵轴performance, 把之前算出来的数据用数组存起来然后画图
    x_points = epoches
    y_points = performance
    plt.plot(x_points, y_points)
    plt.ylabel(title)
    #plt.savefig(title)
    plt.show()
    # pass


def run(config, pip):
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
    optimizer = optim.Adadelta(model.parameters(), lr=config.lr)  # try to find the best output

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
        training_accuracies.append(train_acc)
        training_loss.append(train_loss)
        test_acc, test_loss =test(config, model, device, test_loader)
        """record testing info, Fill your code"""
        testing_accuracies.append((test_acc))
        testing_loss.append(test_loss)
        scheduler.step()
        """update the records, Fill your code"""
        epoches.append(epoch)

    """plotting training performance with the records"""
    plot(epoches, testing_accuracies, "testing_accuracies"+str(config.seed))
    plot(epoches, training_loss, "training_loss"+str(config.seed))

    """plotting testing performance with the records"""
    plot(epoches, testing_accuracies, "testing_accuracies"+str(config.seed))
    plot(epoches, testing_loss, "testing_loss"+str(config.seed))

    if config.save_model:
        torch.save(model.state_dict(), "mnist_cnn.pt")
    pip.send(training_accuracies)
    pip.send(training_loss)
    pip.send(testing_accuracies)
    pip.send(testing_loss)

def plot_mean(result_matrix):
    """
    Read the recorded results.
    Plot the mean results after three runs.
    :return:
    """
    # 需要自己定义record data的函数
    """fill your code"""
    training_accuracies_mean = []
    training_loss_mean = []
    testing_accuracies_mean = []
    testing_loss_mean = []
    epoches = []

    for i in range(len(result_matrix[0][0])):
        mean = 0
        for j in range(len(result_matrix[0])):
            mean += result_matrix[0][j][i] / len(result_matrix[0])
        training_accuracies_mean.append(mean)
    for i in range(len(result_matrix[1][0])):
        mean = 0
        for j in range(len(result_matrix[1])):
            mean += result_matrix[1][j][i] / len(result_matrix[1])
        training_loss_mean.append(mean)
    for i in range(len(result_matrix[2][0])):
        mean = 0
        for j in range(len(result_matrix[2])):
            mean += result_matrix[2][j][i] / len(result_matrix[2])
        testing_accuracies_mean.append(mean)
    for i in range(len(result_matrix[3][0])):
        mean = 0
        for j in range(len(result_matrix[3])):
            mean += result_matrix[3][j][i] / len(result_matrix[3])
        testing_loss_mean.append(mean)
    for i in range(1, len(training_accuracies_mean) + 1):
        epoches.append(i)

    plot(epoches, training_accuracies_mean, "training_accuracies_mean")
    plot(epoches, training_loss_mean, "training_loss_mean")

    plot(epoches, testing_accuracies_mean, "testing_accuracies_mean")
    plot(epoches, testing_loss_mean, "testing_loss_mean")


if __name__ == '__main__':
    pips = []
    processes = []
    final_result = [[], [], [], []]
    files = os.listdir(".")
    #arg = read_args()

    for file in files:
        if ".txt" in file:
            os.remove(file)
            with open(file, "w") as f:
                pass
        if ".yaml" in file:
            # multi-processing 的内容
            arg = args_input()
            arg.config_file = file
            config = load_config(arg)
            pipe_receive, pipe_send = multiprocessing.Pipe(duplex=False)
            pips.append(pipe_receive)
            processes.append(multiprocessing.Process(target=run, args=(config, pipe_send)))
            processes[-1].start()

    """toad training settings"""
    # config = load_config(arg)

    """train model and record results"""
    for pipe in pips:
        final_result[0].append(pipe.recv())
        final_result[1].append(pipe.recv())
        final_result[2].append(pipe.recv())
        final_result[3].append(pipe.recv())
    for process in processes:
        process.join()

    """plot the mean results"""
    plot_mean(final_result)
