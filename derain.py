import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from RNetP import RNetP
from RNet import RNet
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


def main(mod, dataset):
    # Build model
    print('Model: ' + mod + '\n')
    if mod == 'RNetP':
        model = RNetP(channels=3)
        print('Loading model\n')
        model = model.cuda()
        if dataset == 'Rain100L':
            model.load_state_dict(torch.load('.'))
        else:
            model.load_state_dict(torch.load(opt.modelDir))
    else:
        model = DnCNN_Res(channels=3)
        print('Loading model\n')
        model = model.cuda()
        if dataset == 'Rain100L':
            model.load_state_dict(torch.load(opt.modelDir))
        else:
            model.load_state_dict(torch.load(opt.modelDir))


    model.eval()

    # process data
    psnr_test = 0
    ssim_test = 0
    print('Deraining\n')
    for f in os.listdir(opt.rainDir):
        # image
        rain = cv2.imread(os.path.join(opt.rainDir, f)) / 255
        rain = np.transpose(rain, (2, 0, 1))
        rain = np.expand_dims(rain, axis=0)
        rain = torch.Tensor(rain).cuda()

        gt = cv2.imread(os.path.join(opt.gtDir, f)) / 255

        mask = model(rain)
        out = torch.clamp(rain - mask, 0., 1.)
        out = out.data.cpu().numpy().astype(np.float32)
        out = np.transpose(out[0, :, :, :], (1, 2, 0))

        psnr = peak_signal_noise_ratio(out, gt, data_range=1)
        ssim = structural_similarity(out, gt, data_range=1, channel_axis=2)
        psnr_test += psnr
        ssim_test += ssim

        # cv2.imwrite(os.path.join(opt.outDir, f), out*255)
        # print("%s PSNR %f SSIM %f" % (f, psnr, ssim))
    psnr_test /= len(os.listdir(opt.rainDir))
    ssim_test /= len(os.listdir(opt.rainDir))

    print("PSNR on test data %f SSIM on test data %f\n" % (psnr_test, ssim_test))
    return ssim_test, psnr_test


if __name__ == "__main__":
    l = len(os.listdir(opt.modelDir))

    opt.modelDir = os.path.join(dir, 'net' + str(i) + '.pth')
    print(opt.modelDir)
    ssim_test, psnr_test = main()


