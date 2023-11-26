import cv2 as cv
import sys

# 像素取反
def get_img_reserve(imgpath):
    img = cv.imread(imgpath)
    dst = cv.bitwise_not(img)
    savepath = "D:\File\Study\software_engineering\project\web_server\myapp\\res\\2.jpeg"
    cv.imwrite(savepath,dst)
    return savepath

if __name__ == "__main__":
    get_img_reserve(sys.argv[1])