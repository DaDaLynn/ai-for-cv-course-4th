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
        self.conv3 = nn.Conv2d(48, 120, 3)
        self.pool3 = nn.AvgPool2d(2)
        self.relu3 = nn.ReLU(inplace=True)

        # block4 ==> 29
        self.conv4 = nn.Conv2d(120, 120, 3)
        self.pool4 = nn.AvgPool2d(2)
        self.relu4 = nn.ReLU(inplace=True)

        # branch1
        # ==> 14
        self.conv51 = nn.Conv2d(120, 48, 3)
        self.pool51 = nn.AvgPool2d(2)
        self.relu51 = nn.ReLU(inplace=True)

        # ==> 5
        self.conv61 = nn.Conv2d(48, 48, 3)
        self.pool61 = nn.AvgPool2d(2)
        self.relu61 = nn.ReLU(inplace=True)

        # ==> 1
        self.conv71 = nn.Conv2d(48, 48, 3)
        self.pool71 = nn.AvgPool2d(2)
        self.relu71 = nn.ReLU(inplace=True)

        self.fc81 = nn.Linear(48, 120)
        self.relu81 = nn.ReLU(inplace=True)

        self.fc91 = nn.Linear(120, 3)
        self.softmax91 = nn.Softmax()

        # branch2
        # ==> 14
        self.conv52 = nn.Conv2d(120, 48, 3)
        self.pool52 = nn.AvgPool2d(2)
        self.relu52 = nn.ReLU(inplace=True)

        self.fc62 = nn.Linear(48 * 13 * 13, 120)
        self.relu62 = nn.ReLU(inplace=True)

        self.fc72 = nn.Linear(120, 2)
        self.softmax72 = nn.Softmax()


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
        x1 = self.conv51(x)
        x1 = self.pool51(x1)
        x1 = self.relu51(x1)

        x1 = self.conv61(x1)
        x1 = self.pool61(x1)
        x1 = self.relu61(x1)

        x1 = self.conv71(x1)
        x1 = self.pool71(x1)
        x1 = self.relu71(x1)

        x1 = x1.view(-1, 48 * 1 * 1)
        x1 = self.fc81(x1)
        x1 = self.relu81(x1)

        x1 = self.fc91(x1)
        x_classes = self.softmax91(x1)

        # branch2
        x2 = self.conv52(x)
        x2 = self.pool52(x2)
        x2 = self.relu52(x2)

        x2 = x2.view(-1, 48 * 13 * 13)
        x2 = self.fc62(x2)
        x2 = self.relu62(x2)

        x2 = self.fc72(x2)
        x_species = self.softmax72(x2)

        return x_classes, x_species

CLASS =['Mammals', 'Birds']
SPECIES = ['rabbits', 'rats', 'chickens']
if __name__ == '__main__':
    sample = torch.randn(1, 3, 500, 500)
    Net_sample = Net()
    y1, y2 = Net_sample.forward(sample)
    _, pred_class = torch.max(y1.view(-1, 3), 1)
    _, pred_species = torch.max(y2.view(-1, 2), 1)
    print("Classes: {} Species: {}".format(CLASS[pred_class], SPECIES[pred_species]))