import sys

import cv2
import numpy
import numpy as np
from torchvision import transforms
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset
import albumentations as A
from tqdm import tqdm
import pathlib
import change_detection_pytorch as cdp
from change_detection_pytorch.datasets import LEVIR_CD_Dataset, SVCD_Dataset
from change_detection_pytorch.utils.lr_scheduler import GradualWarmupScheduler

DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

model = cdp.Unet(
    encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
    encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
    in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
    classes=2,  # model output channels (number of classes in your datasets)
    siam_encoder=True,  # whether to use a siamese encoder
    fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
)


model_path = './best_modelpng_My.pth'
model.to(DEVICE)
# model.load_state_dict(torch.load(model_path))
model = torch.load(model_path)
model.eval()
img_list = pathlib.Path("./data/mydataset/val/Ab")
img_list = list(img_list.glob("*.png"))
img_list = list(str(x)[21:] for x in img_list)
test_transform = A.Compose([
    A.Resize(256,256),
    A.Normalize(mean=0.5,std=0.5)])
t = A.Compose([
A.Resize(500,500)
])
for filename in img_list:
    img1 = cv2.imread("./data/mydataset/val/Ab" + filename,cv2.IMREAD_UNCHANGED)
    img1 = test_transform(image=img1)
    img1 = img1['image']
    img1 = np.expand_dims(img1, 0)
    img1 = np.expand_dims(img1, 0)
    img1 = torch.Tensor(img1)
    img1 = img1.cuda()

    img2 = cv2.imread("./data/mydataset/val/Bb" + filename,cv2.IMREAD_UNCHANGED)
    img2 = test_transform(image=img2)
    img2 = img2['image']
    img2 = np.expand_dims(img2, 0)
    img2 = np.expand_dims(img2, 0)
    img2 = torch.Tensor(img2)
    img2 = img2.cuda()


    pre = model(img1, img2)
    pre = torch.argmax(pre, dim=1).cpu().data.numpy()
    pre = (pre*225).astype(np.uint8)
    # pre = transforms.ToPILImage(pre[0])
    pre = Image.fromarray(pre[0])
    pre = pre.resize((500,500))
    pre.save("./res" + filename)
    # pre = pre.numpy()
    # cv2.imwrite("./res" + filename, pre[0])