import os
from pathlib import Path
from random import randint
import cv2


dir_rain = Path('data/source/rain')
dir_norain = Path('data/source/norain')

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
        cv2.imwrite("./data/rain/"+img.name, ipt)
        cv2.imwrite("./data/masks/"+img.name, mask)
        cv2.imwrite("./data/norain/" + img.name, gt)

print("done")
