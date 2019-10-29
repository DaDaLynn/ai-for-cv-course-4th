import pandas as pd
import os
from PIL import Image
from torchvision.transforms import transforms
from torch.utils.data import DataLoader
import torch
import random
import matplotlib.pyplot as plt
from my_Network0 import *
from torch import optim
from torch.optim import lr_scheduler
import copy

class MyDataset():
    def __init__(self, root_path, annotation_file, transform):
        self.path = root_path
        self.annotation_file = annotation_file
        if os.path.isfile(os.path.join(root_path, annotation_file)):
            self.file_info = pd.read_csv(annotation_file)
        self.size = len(self.file_info)
        self.transform = transform

    def __len__(self):
        return self.size

    def __getitem__(self, idx):
        if idx > self.size:
            return None
        sample_path = self.file_info['image'][idx]
        image = Image.open(sample_path).convert('RGB')
        if self.transform:
            image = self.transform(image)
        classes = self.file_info['class'][idx]
        species = self.file_info['species'][idx]
        sample = {'image': image, 'class': classes, 'species': species}
        return sample

RootPath = r'D:\Lynn\AI-for-CV\Class_material\project\projectI\code\stage3'
Train_anno = 'Classes_Species_train_annotation.csv'
Val_anno = 'Classes_Species_val_annotation.csv'

train_transform = transforms.Compose([transforms.Resize((500, 500)),
                                      transforms.RandomHorizontalFlip(),
                                      transforms.ToTensor()])

val_transform = transforms.Compose([transforms.Resize((500, 500)),
                                    transforms.ToTensor()])

train_dataset = MyDataset(RootPath, Train_anno, train_transform)

test_dataset = MyDataset(RootPath, Val_anno, val_transform)

train_loader = DataLoader(train_dataset, 32, True)
test_loader = DataLoader(test_dataset)
data_loaders = {'train': train_loader, 'val': test_loader}

device = torch.device("cuda:0" if torch.cuda.is_available() else 'cpu')
print(device)

def visualize_dataset():
    idx = random.randint(0, len(train_dataset))
    sample = train_dataset[idx]
    print("idx = {} image.shape = {} class = {} species = {}".format(idx, sample['image'].shape, sample['class'], sample['species']))
    img = sample['image']
    plt.imshow(transforms.ToPILImage()(img))
    plt.show()

visualize_dataset()

def train_model(model, criterion, optimizer, schduler, num_epoches = 50):
    Loss_list = {'train':[], 'val':[]}
    Accuracy_list = {'train':[], 'val':[]}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0
    
    for epoch in range(num_epoches):
        print('Epoch {}/{}'.format(epoch, num_epoches - 1))
        print('-*' * 10)

        for s in ['train', 'val']:
            if s == 'train':
                model.train()
            else:
                model.eval()

            running_loss = 0.0
            corrects_count = 0.0

            for idx, data in enumerate(data_loaders[s]):
                print(s + ' processing: {}th batch.'.format(idx))
                inputs = data['image'].to(device)
                class_labels = data['class'].to(device)
                species_labels = data['species'].to(device)

                optimizer.zero_grad()

                with torch.set_grad_enabled(s == 'train'):
                    x_classes, x_species = model(inputs)

                    x_classes.view(-1, 2)
                    _, preds_classes = torch.max(x_classes, 1)
                    x_species.view(-1, 3)
                    _, preds_species = torch.max(x_species, 1)

                    loss_class = criterion(x_classes, class_labels)
                    loss_species = criterion(x_species, species_labels)

                    loss = 0.5 * loss_class + 0.5 * loss_species

                    if s == 'train':
                        loss.backward()
                        optimizer.step()
                
                running_loss += loss.item() * inputs.size(0)
                correct_class = (preds_classes == class_labels)
                correct_species = (preds_species == species_labels)
                correct_ret = correct_class & correct_species
                corrects_count += torch.sum(correct_ret)

            epoch_loss = running_loss / len(data_loaders[s].dataset)
            Loss_list[s].append(epoch_loss)

            epoch_acc = corrects_count.double() / len(data_loaders[s].dataset)
            
            Accuracy_list[s].append(100 * epoch_acc)
            print('{} Loss: {:.4f} Acc: {:.2%}'.format(s, epoch_loss, epoch_acc))

            if s == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())
                print('Best val Acc: {:.2%}'.format(best_acc))
    
    model.load_state_dict(best_model_wts)
    torch.save(model.state_dict(), 'best_model.pt')
    print('Best val Acc: {:.2%}'.format(best_acc))
    return model, Loss_list, Accuracy_list

network = Net().to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(network.parameters(), lr=0.001, momentum=0.9)
exp_lr_schduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)
model, Loss_list, Accuracy_list = train_model(network, criterion, optimizer, exp_lr_schduler, num_epoches=10)

x = range(0, 10)
y1 = Loss_list['val']
y2 = Loss_list['train']

plt.plot(x, y1, color="r", linestyle='-', marker='o', linewidth=1, label='val')
plt.plot(x, y2, color='b', linestyle='-', marker='o', linewidth=1, label='train')
plt.legend()
plt.title('train and val loss vs. epoches')
plt.ylabel('loss')
plt.savefig('train and val loss epoches.jpg')
plt.close('all')

y5 = Accuracy_list['train']
y6 = Accuracy_list['val']
plt.plot(x, y5, color='r', linestyle='-', marker='.', linewidth=1, label='train')
plt.plot(x, y6, color='b', linestyle='-', marker='.', linewidth=1, label='val')
plt.legend()
plt.title('train and val accuracy vs. epoches')
plt.ylabel('Accuracy')
plt.savefig('train and val accuracy vs epoches.jpg')
plt.close('all')

CLASS =['Mammals', 'Birds']
SPECIES = ['rabbits', 'rats', 'chickens']
def visualize_model(model):
    model.eval()
    with torch.no_grad():
        for i, data in enumerate(data_loaders['val']):
            inputs = data['image']
            labels_class = data['class'].to(device)

            x_clases = model(inputs.to(device))
            x_classes = x_classes.view(-1, 3)
            _, preds_classes = torch.max(x_clases, 1)

            labels_species = data['species'].to(device)
            x_species = model(inputs.to(device))
            x_species = x_species.view(-1, 2)
            _, preds_species = torch.max(x_species, 1)

            print(inputs.shape)
            plt.imshow(transforms.ToPILImage()(inputs.squeeze(0)))
            plt.title('predicted：classes-{} species-{}\n ground-truth: classes-{} species-{}'.format(CLASS[preds_classes], CLASS[labels_class], SPECIES[preds_species], SPECIES[labels_species]))
            plt.show()

#visualize_model(model)



