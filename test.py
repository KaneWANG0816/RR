import cv2
import os
import argparse
import glob
import numpy as np
import torch
import torch.nn as nn
from torch.autograd import Variable
import matplotlib.pyplot as plt
from model import DnCNN
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN_Test")
parser.add_argument("--nl", type=int, default=17, help="Number of layers")
parser.add_argument("--modelDir", type=str, default="models", help='path of models')
parser.add_argument("--rainDir", type=str, default='./data/test/rain', help='path of rain')
parser.add_argument("--gtDir", type=str, default='./data/test/norain', help='path of ground truth')
parser.add_argument("--outDir", type=str, default='./out', help='path of derain results')
opt = parser.parse_args()


def main():
    # Build model
    print('Loading model ...\n')
    model = DnCNN(channels=3, num_of_layers=opt.nl)
    model = model.cuda()
    model.load_state_dict(torch.load(os.path.join(opt.modelDir, 'net14.pth')))
    model.eval()

    # process data

    print('Loading data info ...\n')
    psnr_test = 0
    ssim_test = 0
    for f in os.listdir(opt.rainDir):
        # image
        rain = cv2.imread(os.path.join(opt.rainDir, f)) / 255
        rain = np.transpose(rain, (2, 0, 1))
        rain = np.expand_dims(rain, axis=0)
        rain = torch.Tensor(rain).cuda()

        gt = cv2.imread(os.path.join(opt.gtDir, f)) / 255

        mask = model(rain)
        out = torch.clamp(rain-mask, 0., 1.)
        out = out.data.cpu().numpy().astype(np.float32)
        out = np.transpose(out[0, :, :, :], (1, 2, 0))

        psnr = peak_signal_noise_ratio(out, gt, data_range=1)
        ssim = structural_similarity(out, gt, data_range=1, channel_axis=2)
        psnr_test += psnr
        ssim_test += ssim

        cv2.imwrite(os.path.join(opt.outDir, f), out*255)
        print("%s PSNR %f SSIM %f" % (f, psnr, ssim))
    psnr_test /= len(os.listdir(opt.rainDir))
    ssim_test /= len(os.listdir(opt.rainDir))
    print("\nPSNR on test data %f SSIM on test data %f" % (psnr_test, ssim_test))


if __name__ == "__main__":
    main()
