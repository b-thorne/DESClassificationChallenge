import torch.nn as nn
import torch.nn.functional as F


def CNNBlock(conv_in, conv_out, kernel_size=3, padding=1, max_pool=2):
    return nn.Sequential(
        nn.Conv2d(conv_in, conv_out, kernel_size=kernel_size, padding=padding),
        nn.BatchNorm2d(conv_out, eps=1e-5, momentum=0.99),
        nn.ReLU(),
        nn.MaxPool2d(max_pool),
    )


class BinaryClassifierCNN(nn.Module):
    def __init__(self):
        super(BinaryClassifierCNN, self).__init__()

        self.block1 = CNNBlock(3, 80)
        self.block2 = CNNBlock(80, 80)
        self.block3 = CNNBlock(80, 80)

        self.fc1 = nn.Flatten()
        self.dropout = nn.Dropout(0.3)
        self.fc2 = nn.Linear(2880, 51)
        self.bn = nn.BatchNorm1d(51, eps=1e-5, momentum=0.99)
        self.fc3 = nn.Linear(51, 1)

    def forward(self, x):
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)

        x = x.view(x.size(0), -1)

        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.bn(x)
        x = self.fc3(x)

        return nn.Sigmoid()(x)
