import sys
import model_run

def get_img_reserve(imgpath1,imgpath2,savepath):
    try:
        model_run.mode_run(imgpath1, imgpath2, savepath)
    except:
        return None
    return 1

if __name__ == "__main__":
    if len(sys.argv) >= 4:
        if get_img_reserve(sys.argv[1],sys.argv[2],sys.argv[3]) != None:
            # print("normal out")
            exit(0)
    print("error out")
    exit(4)
    # get_img_reserve("./data/mydataset/train/A/img1.tif",
    #                 "./data/mydataset/train/B/img1.tif",
    #                 "./res/res.png")