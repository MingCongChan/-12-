import numpy as np
import torch
from PIL import Image
import albumentations as A
import change_detection_pytorch as cdp


def mode_run(img1Path, img2Path, savePath):
    DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'

    model = cdp.Unet(
        encoder_name="resnet34",  # choose encoder, e.g. mobilenet_v2 or efficientnet-b7
        encoder_weights="imagenet",  # use `imagenet` pre-trained weights for encoder initialization
        in_channels=1,  # model input channels (1 for gray-scale images, 3 for RGB, etc.)
        classes=2,  # model output channels (number of classes in your datasets)
        siam_encoder=True,  # whether to use a siamese encoder
        fusion_form='concat',  # the form of fusing features from two branches. e.g. concat, sum, diff, or abs_diff.
    )


    model_path = 'D:\File\Study\software_engineering\project\web_server\myapp\public\python\model\\best_modelpng_My.pth'
    model.to(DEVICE)
    # model.load_state_dict(torch.load(model_path))
    model = torch.load(model_path)
    model.eval()

    test_transform = A.Compose([
        A.Resize(256,256),
        A.Normalize(mean=0.5,std=0.5)])

    img1 = Image.open(img1Path)

    height = img1.height
    width = img1.width

    if img1.mode != "L":
        img1 = img1.convert("L")
    # img1.resize((256,256))
    img1 = np.array(img1)
    img1 = test_transform(image=img1)
    img1 = img1['image']
    img1 = np.expand_dims(img1, 0)
    img1 = np.expand_dims(img1, 0)
    img1 = torch.Tensor(img1)
    img1 = img1.cuda()


    img2 = Image.open(img2Path)
    if img2.mode != "L":
        img2 = img2.convert("L")
    img2 = np.array(img2)
    img2 = test_transform(image=img2)
    img2 = img2['image']
    img2 = np.expand_dims(img2, 0)
    img2 = np.expand_dims(img2, 0)
    img2 = torch.Tensor(img2)
    img2 = img2.cuda()


    pre = model(img1, img2)
    pre = torch.argmax(pre, dim=1).cpu().data.numpy()
    pre = (pre*225).astype(np.uint8)
    pre = Image.fromarray(pre[0])
    pre = pre.resize((width,height))
    pre.save(savePath)
