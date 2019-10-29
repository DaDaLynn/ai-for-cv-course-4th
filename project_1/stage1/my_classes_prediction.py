import pandas as pd
import os
from PIL import Image
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
import random
import matplotlib.pyplot as plot
import copy

from torch import optim
from torch.optim import lr_scheduler
from my_classes_Network import *

Root_Path = r"D:\Lynn\AI-for-CV\Class_material\project\projectI\code"
CLASSES = ["Mammals", "Birds"]

class myDataset():
    def __init__(self, root_path, csvfile, transform=None):
        self.root = root_path
        try:
            self.data_info = pd.read_csv(os.path.join(root_path, csvfile))
        except:
            print("csvfile does not exist")
        self.size = len(self.data_info)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        sample_path = self.data_info["path"][idx]
        try:
            sample = Image.open(sample_path).convert('RGB')
        except:
            print("sample does not exist")
            return None
        
        if(self.transform):
            sample = self.transform(sample)
        label = self.data_info["label"][idx]
        return {"data":sample, "label":label}

train_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                       transforms.RandomHorizontalFlip(),
                                       transforms.ToTensor()
                                       ])

val_transforms = transforms.Compose([transforms.Resize((500, 500)),
                                     transforms.ToTensor()
                                     ])

train_dataset = myDataset(Root_Path, "Classes_train_annotation.csv", train_transforms)
test_dataset  = myDataset(Root_Path, "Classes_val_annotation.csv", val_transforms)

train_dataloader = DataLoader(dataset=train_dataset, batch_size=24, shuffle=True)
test_dataloader  = DataLoader(dataset=test_dataset)
data_loaders = {"train":train_dataloader, "val":test_dataloader}

device = torch.device("cuda: 0" if torch.cuda.is_available() else "cpu")

def visualize_data(dataloader):
    dataset = dataloader.dataset
    idx = random.randint(0, len(dataset))
    sample = dataset[idx]
    print(idx, sample["data"].shape, CLASSES[sample["label"]])
    img = sample["data"]
    plot.imshow(transforms.ToPILImage()(img))
    plot.show()

visualize_data(train_dataloader)

def train(model, criterion, optimizer, schduler, num_epochs=50):
    Loss_list = {'train':[], 'val':[]}
    Accuracy_list_classes = {'train':[], 'val':[]}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-*' * 10)

        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_classes = 0

            for idx,data in enumerate(data_loaders[phase]):
                print(phase+' processing: {}th batch.'.format(idx))
                inputs = data['data'].to(device)
                labels_classes = data['label'].to(device)
                optimizer.zero_grad()

                with torch.set_grad_enabled(phase == 'train'):
                    x_classes = model(inputs)

                    x_classes = x_classes.view(-1, 2)

                    _, preds_classes = torch.max(x_classes, 1)

                    loss = criterion(x_classes, labels_classes)

                    if phase == 'train':
                        loss.backward()
                        optimizer.step()

                running_loss += loss.item() * inputs.size(0)

                corrects_classes += torch.sum(preds_classes == labels_classes)

            epoch_loss = running_loss / len(data_loaders[phase].dataset)
            Loss_list[phase].append(epoch_loss)

            epoch_acc_classes = corrects_classes.double() / len(data_loaders[phase].dataset)
            epoch_acc = epoch_acc_classes

            Accuracy_list_classes[phase].append(100 * epoch_acc_classes)
            print('{} Loss: {:.4f}   Acc_classes: {:.2%}'.format(phase, epoch_loss, epoch_acc_classes))

            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc_classes
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val classes Acc: {:.2%}'.format(best_acc))

    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict, 'best_model.pt')
    print('Best val classes Acc: {:.2%}'.format(best_acc))
    return model, Loss_list, Accuracy_list_classes

network = Net().to(device)
optimizer = optim.SGD(network.parameters(), lr=0.01, momentum=0.9)
criterion = nn.CrossEntropyLoss()
exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1) 
model, Loss_list, Accuracy_list_classes = train(network, criterion, optimizer, exp_lr_scheduler, num_epochs=100)

x = range(0, 100)
y1 = Loss_list["val"]
y2 = Loss_list["train"]

plot.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
plot.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
plot.legend()
plot.title('train and val loss vs. epoches')
plot.ylabel('loss')
plot.savefig("train and val loss epoches.jpg")
plot.close('all')

y5 = Accuracy_list_classes["train"]
y6 = Accuracy_list_classes["val"]
plot.plot(x, y5, color="r", linestyle="-", marker=".", linewidth=1, label="train")
plot.plot(x, y6, color="r", linestyle="-", marker=".", linewidth=1, label="val")
plot.legend()
plot.title('train and val Classes_acc vs. epoches')
plot.ylabel('Classes_accuracy')
plot.savefig("train and val Class_acc vs epoches.jpg")

def visualize_model(model):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loaders['val']):
            inputs = data['data']
            labels_classes = data['label'].to(device)

            x_classes = model(inputs.to(device))
            x_classes = x_classes.view(-1, 2)
            _, preds_classes = torch.max(x_classes, 1)

            print(inputs.shape)
            plot.imshow(transforms.ToPILImage()(inputs.squeeze(0)))
            plot.title('predicted classes: {}\n ground-truth classes:{}'.format(CLASSES[preds_classes], CLASSES[labels_classes]))
            plot.show()
visualize_model(model)
    




