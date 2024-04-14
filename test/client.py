from collections import OrderedDict

import flwr as fl
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import CIFAR10
from torchvision.transforms import Compose, Normalize, ToTensor
from tqdm import tqdm


DEVICE = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# 定义神经网络模型

class Net(nn.Module):
    def __init__(self) -> None:
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 6, 5)
        self.pool = nn.MaxPool2d(2, 2)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1 = nn.Linear(16 * 5 * 5, 120)
        self.fc2 = nn.Linear(120, 84)
        self.fc3 = nn.Linear(84, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool(F.relu(self.conv1(x)))
        x = self.pool(F.relu(self.conv2(x)))
        x = x.view(-1, 16 * 5 * 5)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

# 定义模型训练流程

def train(net, trainloader, epochs):
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    for _ in range(epochs):
        for images, labels in tqdm(trainloader):
            optimizer.zero_grad()
            criterion(net(images.to(DEVICE)), labels.to(DEVICE)).backward()
            optimizer.step()

# 定义模型推理流程

def test(net, testloader):
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    with torch.no_grad():
        for images, labels in tqdm(testloader):
            outputs = net(images.to(DEVICE))
            labels = labels.to(DEVICE)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
    accuracy = correct / len(testloader.dataset)
    return loss, accuracy

# 定义数据集的获取

def load_data():
    """Load CIFAR-10 (training and test set)."""
    trf = Compose([ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    trainset = CIFAR10("./data", train=True, download=True, transform=trf)
    testset = CIFAR10("./data", train=False, download=True, transform=trf)
    return DataLoader(trainset, batch_size=32, shuffle=True), DataLoader(testset)

# 生成模型对象，实际获取训练与测试数据集

net = Net().to(DEVICE)
trainloader, testloader = load_data()

class FlowerClient(fl.client.NumPyClient):
    # 获取本地模型对应的参数

    def get_parameters(self, config):
        return [val.cpu().numpy() for _, val in net.state_dict().items()]

    # 接收模型参数，并更新本地模型

    def set_parameters(self, parameters):
        params_dict = zip(net.state_dict().keys(), parameters)
        state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
        net.load_state_dict(state_dict, strict=True)

    # 本地模型训练，会先调用 set_parameters() 基于收到的全局模型参数更新本地模型

    def fit(self, parameters, config):
        self.set_parameters(parameters)
        train(net, trainloader, epochs=1)
        return self.get_parameters(config={}), len(trainloader.dataset), {}

    # 基于测试数据集进行测试

    def evaluate(self, parameters, config):
        self.set_parameters(parameters)
        loss, accuracy = test(net, testloader)
        return loss, len(testloader.dataset), {"accuracy": accuracy}

# 启动 Flower 客户端

fl.client.start_client(
    server_address="127.0.0.1:8080",
    client=FlowerClient(),
)