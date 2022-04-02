import math
import torch
import torch.nn as nn
import numpy as np
import cv2
from skimage.metrics import peak_signal_noise_ratio, structural_similarity


def transform(img):
    # BGR to RGB
    b, g, r = cv2.split(img)
    img = cv2.merge([r, g, b])
    return img


def weights_init_kaiming(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
    elif classname.find('Linear') != -1:
        nn.init.kaiming_normal(m.weight.data, a=0, mode='fan_in')
    elif classname.find('BatchNorm') != -1:
        # nn.init.uniform(m.weight.data, 1.0, 0.02)
        m.weight.data.normal_(mean=0, std=math.sqrt(2. / 9. / 64.)).clamp_(-0.025, 0.025)
        nn.init.constant_(m.bias.data, 0.0)


def batch_PSNR(img1, img2, data_range):
    img1 = img1.data.cpu().numpy().astype(np.float32)
    img2 = img2.data.cpu().numpy().astype(np.float32)
    PSNR = 0
    for i in range(img1.shape[0]):
        PSNR += peak_signal_noise_ratio(img2[i, :, :, :], img1[i, :, :, :], data_range=data_range)
    return PSNR / img1.shape[0]


def batch_SSIM(img1, img2, data_range):
    img1 = img1.data.cpu().numpy().astype(np.float32)
    img2 = img2.data.cpu().numpy().astype(np.float32)
    SSIM = 0
    for i in range(img1.shape[0]):
        SSIM += structural_similarity(img2[i, :, :, :], img1[i, :, :, :], data_range=data_range, channel_axis=0)
    return SSIM / img1.shape[0]


