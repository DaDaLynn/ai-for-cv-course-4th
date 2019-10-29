import torch
import torch.nn as nn
import torch.nn.functional as F

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        # block1 ==> 249
        self.conv1 = nn.Conv2d(3, 12, 3, 2)
        self.bn1 = nn.BatchNorm2d(12) 
        self.relu1 = nn.ReLU(inplace=True)

        # block2 ==> 123
        self.conv2 = nn.Conv2d(12, 48, 3)
        self.pool2 = nn.AvgPool2d(2)
        self.relu2 = nn.ReLU(inplace=True)

        # block3 ==> 60
        self.conv3 = nn.Conv2d(48, 64, 3)
        self.pool3 = nn.AvgPool2d(2)
        self.relu3 = nn.ReLU(inplace=True)

        # block4 ==> 29
        self.conv4 = nn.Conv2d(64, 64, 3)
        self.pool4 = nn.AvgPool2d(2)
        self.relu4 = nn.ReLU(inplace=True)

        # branch1
        self.fc51 = nn.Linear(64 * 29 * 29, 120)
        self.relu51 = nn.ReLU(inplace=True)

        self.fc61 = nn.Linear(120, 3)
        self.softmax61 = nn.Softmax(dim=1)

        # branch2
        self.fc52 = nn.Linear(64 * 29 * 29, 64)
        self.relu52 = nn.ReLU(inplace=True)

        self.fc62 = nn.Linear(64, 2)
        self.softmax62 = nn.Softmax(dim=1)


    def forward(self, x):
        # block1
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu1(x)

        # block2
        x = self.conv2(x)
        x = self.pool2(x)
        x = self.relu2(x)

        # block3
        x = self.conv3(x)
        x = self.pool3(x)
        x = self.relu3(x)

        # block4
        x = self.conv4(x)
        x = self.pool4(x)
        x = self.relu4(x)

        # branch1
        x1 = x.view(-1, 64 * 29 * 29)
        x1 = self.fc51(x1)
        x1 = self.relu51(x1)

        x1 = self.fc61(x1)
        x_species= self.softmax61(x1)

        # branch2
        x2 = x.view(-1, 64 * 29 * 29)
        x2 = self.fc52(x2)
        x2 = self.relu52(x2)

        x2 = self.fc62(x2)
        x_classes = self.softmax62(x2)

        return x_classes, x_species

CLASS =['Mammals', 'Birds']
SPECIES = ['rabbits', 'rats', 'chickens']
if __name__ == '__main__':
    sample = torch.randn(1, 3, 500, 500)
    Net_sample = Net()
    y1, y2 = Net_sample.forward(sample)
    _, pred_class = torch.max(y1.view(-1, 2), 1)
    _, pred_species = torch.max(y2.view(-1, 3), 1)
    print("Classes: {} Species: {}".format(CLASS[pred_class], SPECIES[pred_species]))