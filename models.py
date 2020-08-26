# Authors: Sebastian Szyller, Buse Gul Atli
# Copyright 2020 Secure Systems Group, Aalto University, https://ssg.aalto.fi
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


import torch
import torch.nn as nn
from torch.nn import functional as F


class MNIST_L2(nn.Module):
    def __init__(self, dropout=0.0):
        super(MNIST_L2, self).__init__()

        self.dropout = dropout

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 10)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


class MNIST_L2_DRP03(nn.Module):
    def __init__(self, dropout=0.3):
        super(MNIST_L2_DRP03, self).__init__()

        self.dropout = dropout

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 10)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


class MNIST_L2_DRP05(nn.Module):
    def __init__(self, dropout=0.5):
        super(MNIST_L2_DRP05, self).__init__()

        self.dropout = dropout

        self.conv1 = nn.Conv2d(1, 32, kernel_size=5)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=5)

        self.relu = nn.ReLU(True)
        self.pool = nn.MaxPool2d(kernel_size=2)
        self.fc1 = nn.Linear(64 * 4 * 4, 10)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.relu(self.pool(self.conv1(x)))
        x = self.relu(self.pool(self.conv2(x)))
        x = x.view(-1, 64 * 4 * 4)
        x = self.dropout(x)
        x = self.fc1(x)

        return F.log_softmax(x, dim=1)


class MNIST_L5(nn.Module):
    def __init__(self, dropout=0.0):
        super(MNIST_L5, self).__init__()

        self.dropout = dropout

        self.block = nn.Sequential(
            nn.Conv2d(1, 32, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128 * 5**2 , 200)
        self.fc2 = nn.Linear(200, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        out = self.block(x)
        out = out.view(-1,  128 * 5**2)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return F.log_softmax(out,1)


class MNIST_L5_DRP03(nn.Module):
    def __init__(self, dropout=0.3):
        super(MNIST_L5_DRP03, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(1, 32, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128 * 5**2, 200)
        self.fc2 = nn.Linear(200, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        out = self.block(x)
        out = out.view(-1,  128 * 5**2)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return F.log_softmax(out, 1)


class MNIST_L5_DRP05(nn.Module):
    def __init__(self, dropout=0.3):
        super(MNIST_L5_DRP05, self).__init__()

        self.block = nn.Sequential(
            nn.Conv2d(1, 32, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128 * 5**2, 200)
        self.fc2 = nn.Linear(200, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        out = self.block(x)
        out = out.view(-1,  128 * 5**2)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        out = self.dropout(out)
        out = self.fc2(out)
        return F.log_softmax(out, 1)


class MNIST_L5_with_latent(nn.Module):
    def __init__(self, dropout=0.0):
        super(MNIST_L5_with_latent, self).__init__()

        self.dropout = dropout

        self.block = nn.Sequential(
            nn.Conv2d(1, 32, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(32, 64, 2),
            nn.MaxPool2d(2),
            nn.ReLU(),
            nn.Conv2d(64, 128, 2),
            nn.ReLU(),
        )
        self.fc1 = nn.Linear(128 * 5**2 , 200)
        self.fc2 = nn.Linear(200, 10)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, x):
        x = self.dropout(x)
        out = self.block(x)
        out = out.view(-1,  128 * 5**2)
        out = self.dropout(out)
        out = self.relu(self.fc1(out))
        latent = out
        out = self.dropout(out)
        out = self.fc2(out)
        logit = out
        return F.log_softmax(out,1), [latent, logit]


class CIFAR10_BASE(nn.Module):
    def __init__(self, dropout=0.0):
        super(CIFAR10_BASE, self).__init__()

        self.dropout = dropout
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),  # 0.05

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=0.1),  # 0.1
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=0.1),  # 0.1
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x


class CIFAR10_BASE_DRP03(nn.Module):
    def __init__(self, dropout=0.3):
        super(CIFAR10_BASE_DRP03, self).__init__()

        self.dropout = dropout
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),  # 0.05

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x


class CIFAR10_BASE_DRP05(nn.Module):
    def __init__(self, dropout=0.5):
        super(CIFAR10_BASE_DRP05, self).__init__()

        self.dropout = dropout
        self.conv_layer = nn.Sequential(

            # Conv Layer block 1
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Conv Layer block 2
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=128, out_channels=128, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Dropout2d(p=0.05),  # 0.05

            # Conv Layer block 3
            nn.Conv2d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
        )

        self.fc_layer = nn.Sequential(
            nn.Dropout(p=self.dropout),
            nn.Linear(4096, 1024),
            nn.ReLU(inplace=True),
            nn.Linear(1024, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(p=self.dropout),
            nn.Linear(512, 10)
        )

    def forward(self, x):
        x = self.conv_layer(x)
        x = x.view(x.size(0), -1)
        x = self.fc_layer(x)

        return x


def load_state(model, filename):
    state_dict = torch.load(filename)
    print("Loading state from: {}".format(filename))
    model.load_state_dict(state_dict)
    return model


def save_state(model, filename):
    torch.save(model.state_dict(), filename)
    print("Model saved to:", filename)
