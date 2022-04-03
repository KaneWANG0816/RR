import os
from pathlib import Path
from random import randint
import cv2

def datasetCrop():
    dir_rain = Path('data/rain100L_source/train/rain')
    dir_norain = Path('data/srain100L_ource/train/norain')
    dir_rain_test = Path('data/rain100L_source/test/rain')
    dir_norain_test = Path('data/rain100L_source/test/norain')

    print("Generating rain_masks")
    cropSize = [256, 256]

    with os.scandir(dir_rain) as imgs:
        for img in imgs:
            rain = cv2.imread(str(dir_rain) + '/' + img.name)
            norain = cv2.imread(str(dir_norain) + '/' + img.name)
            mask = rain - norain
            h = mask.shape[0]
            w = mask.shape[1]
            nh = randint(0, h - cropSize[0])
            nw = randint(0, w - cropSize[1])
            ipt = rain[nh:nh+cropSize[0], nw:nw+cropSize[1],:]
            gt = norain[nh:nh + cropSize[0], nw:nw + cropSize[1], :]
            mask = mask[nh:nh+cropSize[0], nw:nw+cropSize[1],:]
            cv2.imwrite("./data/rain100L_crop/train/rain/"+img.name, ipt)
            cv2.imwrite("./data/rain100L_crop/train/masks/"+img.name, mask)
            cv2.imwrite("./data/rain100L_crop/train/norain/" + img.name, gt)

    print("Generating test dataset")
    with os.scandir(dir_rain_test) as imgs:
        for img in imgs:
            rain = cv2.imread(str(dir_rain_test) + '/' + img.name)
            norain = cv2.imread(str(dir_norain_test) + '/' + img.name)
            h = mask.shape[0]
            w = mask.shape[1]
            nh = randint(0, h - cropSize[0])
            nw = randint(0, w - cropSize[1])
            ipt = rain[nh:nh+cropSize[0], nw:nw+cropSize[1],:]
            gt = norain[nh:nh + cropSize[0], nw:nw + cropSize[1], :]
            cv2.imwrite("./data/rain100L_crop/test/rain/"+img.name, ipt)
            cv2.imwrite("./data/rain100L_crop/test/norain/" + img.name, gt)
    print("done")

def datasetRotate():
    dir_rain = Path('data/rain100L_source/train/rain')
    dir_norain = Path('data/rain100L_source/train/norain')
    dir_rain_test = Path('data/rain100L_source/test/rain')
    dir_norain_test = Path('data/rain100L_source/test/norain')

    print("Generating rain_masks")
    imgSize = [321,481]

    with os.scandir(dir_rain) as imgs:
        for img in imgs:
            rain = cv2.imread(str(dir_rain) + '/' + img.name)
            norain = cv2.imread(str(dir_norain) + '/' + img.name)
            mask = rain - norain
            if rain.shape[0]!=imgSize[0]:
                rain = cv2.rotate(rain, cv2.cv2.ROTATE_90_CLOCKWISE)
                norain = cv2.rotate(norain, cv2.cv2.ROTATE_90_CLOCKWISE)
                mask = cv2.rotate(mask, cv2.cv2.ROTATE_90_CLOCKWISE)

            cv2.imwrite("./data/rain100L/train/rain/" + img.name, rain)
            cv2.imwrite("./data/rain100L/train/masks/" + img.name, mask)
            cv2.imwrite("./data/rain100L/train/norain/" + img.name, norain)

    print("Generating test dataset")
    with os.scandir(dir_rain_test) as imgs:
        for img in imgs:
            rain = cv2.imread(str(dir_rain_test) + '/' + img.name)
            norain = cv2.imread(str(dir_norain_test) + '/' + img.name)
            if rain.shape[0] != imgSize[0]:
                rain = cv2.rotate(rain, cv2.cv2.ROTATE_90_CLOCKWISE)
                norain = cv2.rotate(norain, cv2.cv2.ROTATE_90_CLOCKWISE)
            cv2.imwrite("./data/rain100L/test/rain/" + img.name, rain)
            cv2.imwrite("./data/rain100L/test/norain/" + img.name, norain)
    print("done")


if __name__=="__main__":
    datasetRotate()
    # datasetCrop()
