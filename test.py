import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
from models import DnCNN
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--nl", type=int, default=17, help="Number of layers")
parser.add_argument("--modelDir", type=str, default="models", help='path of models')
parser.add_argument("--rainDir", type=str, default='./data/test/rain', help='path of rain')
parser.add_argument("--gtDir", type=str, default='./data/test/norain', help='path of ground truth')
opt = parser.parse_args()


def main():
    # Build model
    print('Loading model ...\n')
    model = DnCNN(channels=3, num_of_layers=opt.nl)
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.modelDir, 'net.pth')))
    model.eval()
    # load data info
    print('Loading data info ...\n')

    # process data
    psnr_test = 0
    for f in os.listdir(opt.rainDir):
        # image
        rain = cv2.imread(os.path.join(opt.rainDir, f))
        rain = np.transpose(rain, (2, 0, 1))
        rain = np.expand_dims(rain, axis=0)
        rain = torch.Tensor(rain / 255).cuda()
        gt = cv2.imread(os.path.join(opt.gtDir, f)) / 255

        with torch.no_grad(): # this can save much memory
            out = torch.clamp(rain-model(rain), 0., 1.)

        out = out.data.cpu().numpy().astype(np.float32)
        out = np.transpose(out, (0, 2, 3, 1))[0, :, :, :]
        psnr = peak_signal_noise_ratio(out, gt, data_range=1)
        psnr_test += psnr
        # cv2.imshow("G",norain)
        # cv2.waitKey()
        # cv2.imshow("O", out)
        # cv2.waitKey()
        print("%s PSNR %f" % (f, psnr))
    psnr_test /= len(os.listdir(opt.rainDir))
    print("\nPSNR on test data %f" % psnr_test)

if __name__ == "__main__":
    main()
