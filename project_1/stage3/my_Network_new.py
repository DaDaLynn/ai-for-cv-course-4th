import torch.nn as nn
import torch
import torch.nn.functional as F 

class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 3, 3)
        self.maxpool1 = nn.MaxPool2d(kernel_size=2)
        self.relu1 = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(3, 6, 3)
        self.maxpool2 = nn.MaxPool2d(kernel_size=2)
        self.relu2 = nn.ReLU(inplace=True)

        self.fc1 = nn.Linear(6 * 123 * 123, 150)
        self.relu3 = nn.ReLU(inplace=True)

        self.drop = nn.Dropout2d()

        self.fc2 = nn.Linear(150, 3)
        self.fc3 = nn.Linear(150, 2)
        self.softmax1 = nn.Softmax(dim=1)

    def forward(self, x):
        x = self.conv1(x)
        x = self.maxpool1(x)
        x = self.relu1(x)

        x = self.conv2(x)
        x = self.maxpool2(x)
        x = self.relu2(x)

        x = x.view(-1, 6 * 123 * 123)
        x = self.fc1(x)
        x = self.relu3(x)

        x = F.dropout(x, training=self.training)

        x_species = self.fc2(x)
        x_species = self.softmax1(x_species)

        x_classes = self.fc3(x)
        x_classes = self.softmax1(x_classes)
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