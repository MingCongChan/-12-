# -*- coding: utf-8 -*-
import sys

import torch.nn as nn
import torch
from tqdm import tqdm


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None, **kwargs):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                               kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                               kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None,
                 groups=1, width_per_group=64):
        super(Bottleneck, self).__init__()

        width = int(out_channel * (width_per_group / 64.)) * groups

        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=width,
                               kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=width, out_channels=width, groups=groups,
                               kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(width)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=width, out_channels=out_channel*self.expansion,
                               kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self,
                 block,
                 blocks_num,
                 num_classes=1000,
                 include_top=True,
                 groups=1,
                 width_per_group=64):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64

        self.groups = groups
        self.width_per_group = width_per_group

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                               padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.fc2 = nn.Conv2d(in_channels=128,out_channels=11,kernel_size=1)
        self.fc3 = nn.Conv2d(in_channels=256,out_channels=11,kernel_size=1)
        self.fc4 = nn.Conv2d(in_channels=512,out_channels=11,kernel_size=1)
        self.ct2_1 = nn.ConvTranspose2d(in_channels=11,out_channels=11,kernel_size=8,stride=8)
        self.ct3_1 = nn.ConvTranspose2d(in_channels=11, out_channels=11, kernel_size=3, stride=17)
        self.ct4_1 = nn.ConvTranspose2d(in_channels=11, out_channels=11, kernel_size=8, stride=36)
        self.ct2_2 = nn.ConvTranspose2d(in_channels=11, out_channels=11, kernel_size=8, stride=8)
        self.ct3_2 = nn.ConvTranspose2d(in_channels=11, out_channels=11, kernel_size=3, stride=17)
        self.ct4_2 = nn.ConvTranspose2d(in_channels=11, out_channels=11, kernel_size=8, stride=36)
        self.o1 = nn.Conv2d(in_channels=22,out_channels=2,kernel_size=1)
        self.o2 = nn.Conv2d(in_channels=22,out_channels=2,kernel_size=1)
        self.o3 = nn.Conv2d(in_channels=22,out_channels=2,kernel_size=1)

        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # output size = (1, 1)
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel,
                            channel,
                            downsample=downsample,
                            stride=stride,
                            groups=self.groups,
                            width_per_group=self.width_per_group))
        self.in_channel = channel * block.expansion

        for _ in range(1, block_num):
            layers.append(block(self.in_channel,
                                channel,
                                groups=self.groups,
                                width_per_group=self.width_per_group))

        return nn.Sequential(*layers)

    def forward_once(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x2 = self.layer2(x)
        x3 = self.layer3(x2)
        x4 = self.layer4(x3)

        x2 = self.fc2(x2)
        x3 = self.fc3(x3)
        x4 = self.fc4(x4)

        # if self.include_top:
        #     x = self.avgpool(x)
        #     x = torch.flatten(x, 1)
        #     x = self.fc(x)

        return x2,x3,x4

    def forward(self,x,y):
        x1,x2,x3 = self.forward_once(x)
        y1,y2,y3 = self.forward_once(y)
        x1 = self.ct2_1(x1)
        x2 = self.ct3_1(x2)
        x3 = self.ct4_1(x3)
        y1 = self.ct2_2(y1)
        y2 = self.ct3_2(y2)
        y3 = self.ct4_2(y3)
        n1 = torch.cat((x1, y1),1)
        n2 = torch.cat((x2, y2), 1)
        n3 = torch.cat((x3, y3), 1)
        n1 = self.o1(n1)
        n2 = self.o1(n2)
        n3 = self.o1(n3)
        # a = torch.cat((n1,n2,n3),dim=1)
        # res = torch.sum(a,dim=1)
        res = n1 + n2 + n3
        res = torch.softmax(res,dim=1)

        return res


def resnet34(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet34-333f7ec4.pth
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet50(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet50-19c8e357.pth
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnet101-5d3b4d8f.pth
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)


def resnext50_32x4d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth
    groups = 32
    width_per_group = 4
    return ResNet(Bottleneck, [3, 4, 6, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)


def resnext101_32x8d(num_classes=1000, include_top=True):
    # https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth
    groups = 32
    width_per_group = 8
    return ResNet(Bottleneck, [3, 4, 23, 3],
                  num_classes=num_classes,
                  include_top=include_top,
                  groups=groups,
                  width_per_group=width_per_group)



from torch.utils.data.dataset import Dataset
from torch.utils.data.dataloader import DataLoader
from torchvision.transforms import transforms
from pathlib import Path
from PIL import Image
import torchvision
import torch.optim as optim
import torch
import torch.nn as nn
from torchvision import transforms, datasets
import matplotlib.pyplot as plt
import numpy as np
import cv2

class MyDataset(Dataset):
    def __init__(self, root, transform=None):
        self.transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Resize(224,antialias=True),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            transforms.CenterCrop(224),  # 从图片中间切出224*224的图片
            # transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
            ## 需要对OpenCV读取的图像进行归一化处理才能与PIL读取的图像一致
            # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 常用标准化
        ])
        self.transform2 = transforms.Compose([

            transforms.Resize(224,antialias=True),  # 缩放图片(Image)，保持长宽比不变，最短边为224像素
            transforms.CenterCrop(224),  # 从图片中间切出224*224的图片
            # transforms.ToTensor(), # 将图片(Image)转成Tensor，归一化至[0, 1]
            ## 需要对OpenCV读取的图像进行归一化处理才能与PIL读取的图像一致
            # transforms.Normalize(mean=[.5, .5, .5], std=[.5, .5, .5]) # 标准化至[-1, 1]，规定均值和标准差
            # transforms.Normalize(mean=[0.485], std=[0.229])  # 常用标准化
            transforms.ToTensor(),
        ])
        self.root = root
        images_path = Path(root + '/A')

        images_list = list(images_path.glob('*.png'))  # list(images_path.glob('*.png'))
        length_root = len(root)
        images_list_str = [str(x)[length_root:] for x in images_list]
        self.images = images_list_str

    def __getitem__(self, item):
        image_path = self.images[item]
        # cv2.setNumThreads(0)
        # # image = cv2.imread(image_path) # 读到的是BGR数据，该方法不能读取带中文路径的图像数据，下行则可读取中文路径。
        # image1 = cv2.imdecode(np.fromfile(image_path + '/A', dtype=np.uint8), 1)  # 1：彩色；2：灰度
        # image2 = cv2.imdecode(np.fromfile(image_path + '/B', dtype=np.uint8), 1)
        # lab = cv2.imdecode(np.fromfile(image_path + '/label', dtype=np.uint8), 1)
        # image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2RGB)  # 转化为RGB，也可以用img = img[:, :, (2, 1, 0)]
        # image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2RGB)
        # lab = cv2.cvtColor(lab, cv2.COLOR_BGR2RGB)
        # # 这时的image是H,W,C的顺序，因此下面需要转化为：C, H, W
        # # image = torch.from_numpy(image).permute(2, 0, 1)
        # image1 = torch.from_numpy(image1).permute(2, 0, 1) / 255  # 归一化[0, 1]才能与PIL读取的数据一致
        # image2 = torch.from_numpy(image2).permute(2, 0, 1) / 255
        # lab = torch.from_numpy(lab).permute(2, 0, 1) / 255

        image1 = Image.open('data\\A' + image_path)
        image2 = Image.open('data\\B' + image_path)
        lab = Image.open('data\\label' + image_path)
        image1 = self.transform(image1)
        image2 = self.transform(image2)

        image_Image = transforms.ToTensor()(lab)
        lab2 = transforms.ToPILImage()(1 - image_Image)
        lab = self.transform2(lab)
        lab2 = self.transform2(lab2)
        return image1, image2, lab , lab2

    def __len__(self):
        return len(self.images)




device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print("using {} device.".format(device))

train_dataset = MyDataset('./data')
train_loader = DataLoader(train_dataset, batch_size=1,shuffle=True, num_workers=0)
validate_dataset = MyDataset('./data')
validate_loader = DataLoader(validate_dataset, batch_size=4,shuffle=False, num_workers=0)


train_num = len(train_dataset)
val_num = len(validate_dataset)

dataloaders_dict = {'train': train_loader, 'val': validate_loader}

def imshow(img):
    plt.figure(figsize=(8,8))
    img = img / 2 + 0.5     # 转换到 [0,1] 之间
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()


net = resnet34()
net.to(device)
# define loss function
loss_function = nn.CrossEntropyLoss()
# loss_function = nn.NLLLoss()

# construct an optimizer
params = [p for p in net.parameters() if p.requires_grad]
optimizer = optim.Adam(params, lr=0.0001)

epochs = 3
best_acc = 0.0
train_steps = len(train_loader)
# image1,image2, lab = next(iter(train_loader))
# imshow(torchvision.utils.make_grid(net(image1.to(device),image2.to(device)).cpu()))

for epoch in range(epochs):
    # train
    net.train()
    running_loss = 0.0
    train_bar = tqdm(train_loader, file=sys.stdout)
    for step, data in enumerate(train_bar):
        image1, image2,labels,lab2 = data
        # imshow(torchvision.utils.make_grid(image1))
        # imshow(torchvision.utils.make_grid(image2))
        # imshow(torchvision.utils.make_grid(labels))
        optimizer.zero_grad()
        lab = torch.cat((labels,lab2),dim=1)
        logits = net(image1.to(device),image2.to(device))
        loss = loss_function(logits, lab.to(device))
        loss.backward()
        optimizer.step()

        # print statistics
        running_loss += loss.item()

        train_bar.desc = "train epoch[{}/{}] loss:{:.3f}".format(epoch + 1,
                                                                  epochs,
                                                                  loss)


net.load_state_dict(torch.load('state_dict_model20.pth'))
for i in range(1,10,1):
    image1,image2, lab,lab2 = next(iter(train_loader))

    x = net(image1.to(device),image2.to(device))[0][0].cpu()
    x = torch.where(x>0.8,torch.tensor(1.000e+00),torch.tensor(-1))
    imshow(torchvision.utils.make_grid(x))
    imshow(torchvision.utils.make_grid(lab2))
    imshow(torchvision.utils.make_grid(lab))
# torch.save(net.state_dict(),'state_dict_model20.pth')
print('Finished Training')
