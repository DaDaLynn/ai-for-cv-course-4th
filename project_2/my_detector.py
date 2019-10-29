from __future__ import print_function
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision import datasets, transforms


import argparse
import os
import numpy as np
import runpy
import matplotlib.pyplot as plt

from PIL import Image, ImageDraw
import cv2
from my_data import get_train_test_set

class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()

        # convolution part
        self.conv1_1 = nn.Conv2d(1, 8, 5, 2, 0)
        self.conv2_1 = nn.Conv2d(8, 16, 3, 1, 0)
        self.conv2_2 = nn.Conv2d(16, 16, 3, 1, 0)
        self.conv3_1 = nn.Conv2d(16, 24, 3, 1, 0)
        self.conv3_2 = nn.Conv2d(24, 24, 3, 1, 0)
        self.conv4_1 = nn.Conv2d(24, 40, 3, 1, 1)
        self.conv4_2 = nn.Conv2d(40, 80, 3, 1, 1)

        # inner product part
        self.ip1 = nn.Linear(4 * 4 * 80, 128)
        self.ip2 = nn.Linear(128, 128)
        self.ip3 = nn.Linear(128, 42)

        # prelu part
        self.relu_conv1_1 = nn.PReLU()
        self.relu_conv2_1 = nn.PReLU()
        self.relu_conv2_2 = nn.PReLU()
        self.relu_conv3_1 = nn.PReLU()
        self.relu_conv3_2 = nn.PReLU()
        self.relu_conv4_1 = nn.PReLU()
        self.relu_conv4_2 = nn.PReLU()
        self.relu_ip1 = nn.PReLU()
        self.relu_ip2 = nn.PReLU()

        # pool part
        self.avg_pool = nn.AvgPool2d(2, 2, ceil_mode=True)

    def forward(self, x):
        # block 1
        # print('x input shape: ', x.shape)
        x = self.avg_pool(self.relu_conv1_1(self.conv1_1(x)))
        # print('x after block1 and pool shape should be 32x8x27x27: ', x.shape)     # good
        # block 2
        x = self.relu_conv2_1(self.conv2_1(x))
        # print('b2: after conv2_1 and prelu shape should be 32x16x25x25: ', x.shape) # good
        x = self.avg_pool(self.relu_conv2_2(self.conv2_2(x)))
        # print('b2: after conv2_2 and prelu and pool shape should be 32x16x12x12: ', x.shape) # good
        # block 3
        x = self.relu_conv3_2(self.conv3_1(x))
        # print('b3: after conv3_1 and pool shape should be 32x24x10x10: ', x.shape)
        x = self.avg_pool(self.relu_conv3_2(self.conv3_2(x)))
        # print('b3: after conv3_2 and pool shape should be 32x24x4x4: ', x.shape)
        # block 4
        x = self.relu_conv4_1(self.conv4_1(x))
        # print('x after conv4_1 and pool shape should be 32x40x4x4: ', x.shape)

        # points branch
        ip3 = self.relu_conv4_2(self.conv4_2(x))
        # print('pts: ip3 after conv4_2 and pool shape should be 32x80x4x4: ', ip3.shape)
        ip3 = ip3.view(-1, 4 * 4 * 80)
        # print('ip3 flatten shape should be 32x1280: ', ip3.shape)
        ip3 = self.relu_ip1(self.ip1(ip3))
        # print('ip3 after ip1 shape should be 32x128: ', ip3.shape)
        ip3 = self.relu_ip2(self.ip2(ip3))
        # print('ip3 after ip2 shape should be 32x128: ', ip3.shape)
        ip3 = self.ip3(ip3)
        # print('ip3 after ip3 shape should be 32x42: ', ip3.shape)
        return ip3

train_losses = []
valid_losses = []

def train(args, train_loader, valid_loader, model, criterion, optimizer, device):
    # save model
    if args.save_model:
        if not os.path.exists(args.save_directory):
            os.makedirs(args.save_directory)

    epoch = args.epochs
    pts_criterion = criterion

    for epoch_id in range(epoch):
        print('Epoch {}/{}'.format(epoch_id, epoch - 1))

        if epoch_id < 20:
            Runoptimizer = optimizer[0]
        else:
            Runoptimizer = optimizer[1]
        # monitor training loss
        train_loss = 0.0
        valid_loss = 0.0

        # train the model
        model.train()
        for batch_idx, batch in enumerate(train_loader):
            img = batch['image']
            landmark = batch['landmarks']

            # ground truth
            input_img = img.to(device)
            target_pts = landmark.to(device)
                
            # clear the gradients of all optimized variables
            Runoptimizer.zero_grad()

            # get output
            output_pts = model(input_img)

            # get loss
            loss = pts_criterion(output_pts, target_pts)

            train_loss  += loss.item() * input_img.size(0)
            print('{:d} loss: {:.6f} train_loss: {:.6f}'.format(batch_idx, loss, train_loss))

            # do BP automatically
            loss.backward()
            Runoptimizer.step()
        
        epoch_loss = train_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        print('train: pts_loss: {:.6f}'.format(epoch_loss))

        model.eval()
        with torch.no_grad():
            for valid_batch_idx, batch in enumerate(valid_loader):
                valid_img = batch['image']
                landmark = batch['landmarks']

                input_img = valid_img.to(device)
                target_pts = landmark.to(device)

                output_pts = model(input_img)

                loss = pts_criterion(output_pts, target_pts)

                valid_loss  += loss.item() * valid_img.size(0)
        epoch_loss = valid_loss / len(train_loader.dataset)
        valid_losses.append(epoch_loss)
        print('Valid: pts_loss: {:.6f}'.format(epoch_loss))
        print('======================================================')
        # save model
        if args.save_model:
            saved_model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(epoch_id) + '.pt')
            torch.save(model.state_dict(), saved_model_name)
    return loss, 0.5

def main_test():
    parser = argparse.ArgumentParser(description='Detector')
    parser.add_argument('--batch-size', type=int, default=32, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=12, metavar='N',
                        help='input batch size for testing (default: 100)')
    parser.add_argument('--epochs', type=int, default=100, metavar='N',
                        help='number of epochs to train (default: 100)')
    parser.add_argument('--lr', type=float, default=0.00001, metavar='LR',
                        help='learning rate (default: 0.001)')
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M',
                        help='SGD momentum (default: 0.5)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=20, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--save-model', action='store_true', default=True,
                        help='save the current Model')
    parser.add_argument('--save-directory', type=str, default='train_models_NoNorm',
                        help='learnt models are saving here')
    parser.add_argument('--phase', type=str, default='Train',  #Train/train, Predict/predict, Finetune/finetune
                        help='training, predicting or finetuning')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    # For single GPU
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if use_cuda else "cpu")
    # For multi GPUs, nothing need to change here
    kwargs = {'num_workers': 1, 'pin_memory': True} if use_cuda else {}

    print('===> Loading Datasets')
    train_set, test_set = get_train_test_set()
    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True)
    valid_loader = torch.utils.data.DataLoader(test_set, batch_size=args.test_batch_size)
    predict_loader = torch.utils.data.DataLoader(test_set)

    print('===> Building Models')
    # For single GPU
    model = Net().to(device)

    criterion_pts = nn.MSELoss()
    #optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)
    optimizer = [optim.Adam(model.parameters(), lr=args.lr), optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum)]

    if args.phase == 'Train' or args.phase == 'train':
        print('===> Start Training')
        _, _ = train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device)
        print('===================================================')
        x = range(0, args.epochs)
        y1 = valid_losses
        y2 = train_losses

        plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
        plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
        plt.legend()
        plt.title('train and val loss vs. epoches')
        plt.ylabel('loss')
        plt.savefig("train and val loss vs epoches.jpg")
        plt.close('all') # 关闭图 0
    elif args.phase == 'Test' or args.phase == 'test':
        print('===> Test')
        model.load_state_dict(torch.load(model_name))
        model.eval()
        test_loss = 0
        with torch.no_grad():
            for data in valid_loader:
                test_img = data['image']
                landmark = data['landmarks']

                input_img = test_img.to(device)
                target_pts = landmark.to(device)

                output_pts = model(input_img)

                loss = criterion_pts(output_pts, target_pts)

                test_loss  += loss.item() * test_img.size(0)
        avg_loss = test_loss / len(valid_loader.dataset)
        print('avage loss of the network : {:.6f}'.format(avg_loss))
    elif args.phase == 'Finetune' or args.phase == 'finetune':
        print('===> Finetune')
        model_name = 'detector_epoch_99.pt'
        model.load_state_dict(torch.load(model_name))
        train_lossed, valid_losses = \
            train(args, train_loader, valid_loader, model, criterion_pts, optimizer, device)
        print('===================================================')
        x = range(0, args.epochs)
        y1 = valid_losses
        y2 = train_losses

        plt.plot(x, y1, color="r", linestyle="-", marker="o", linewidth=1, label="val")
        plt.plot(x, y2, color="b", linestyle="-", marker="o", linewidth=1, label="train")
        plt.legend()
        plt.title('train and val loss vs. epoches')
        plt.ylabel('loss')
        plt.savefig("train and val loss vs epoches.jpg")
        plt.close('all') # 关闭图 0
    elif args.phase == 'Predict' or args.phase == 'predict':
        print('===> Predict')
        root_path = "D:\\Lynn\\AI-for-CV\\Class_material\\project\\projectII_face_keypoints_detection\\data\\I\\I\\"
        model_name = os.path.join(args.save_directory, 'detector_epoch' + '_' + str(args.epochs - 1) + '.pt')
        model.load_state_dict(torch.load(model_name))
        model.eval()
        with torch.no_grad():
            data_file = 'test.txt'
            with open(data_file) as f:
                lines = f.readlines()
                for line in lines:
                    line_parts = line.strip().split()
                    img_name = line_parts[0]
                    rect = list(map(int, list(map(float, line_parts[1:5]))))
                    img = Image.open(root_path + img_name).convert('L')     
                    img_crop = img.crop(tuple(rect))
                    img_crop = np.asarray(img_crop.resize((112, 112), Image.BILINEAR),dtype=np.float32)
                    mean = np.mean(img_crop)
                    std = np.std(img_crop)
                    input_img = (img_crop - mean) / (std + 0.0000001)
                    input_img = np.expand_dims(input_img, axis=0)
                    input_img = np.expand_dims(input_img, axis=0)
                    landmark = model(torch.from_numpy(input_img)).numpy()[0, :]
                    
                    ## 请画出人脸crop以及对应的landmarks
                    # please complete your code under this blank
                    img = Image.fromarray(img_crop)
                    draw = ImageDraw.Draw(img)
                    for i in range(np.size(landmark) // 2):
                        draw.point((landmark[2*i], landmark[2*i+1]), 'rgb(255, 0, 0)')
                    img.show()

if __name__ == '__main__':
    main_test()