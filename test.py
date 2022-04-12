import argparse
import os
from matplotlib import pyplot as plt
from RNetP import RNetP
from RNet import RNet
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Model_Test")
parser.add_argument("--nl", type=int, default=17, help="Number of layers")
parser.add_argument("--modelDir", type=str, default="./models/rain100H_tri", help='path of model')
parser.add_argument("--rainDir", type=str, default='./data/rain100H/test/rain', help='path of rain')
parser.add_argument("--gtDir", type=str, default='./data/rain100H/test/norain', help='path of ground truth')
parser.add_argument("--outDir", type=str, default='./out/rain100H_Res', help='path of derain results')

opt = parser.parse_args()


def main():
    # Build model
    # model = RNet(channels=3)
    model = RNetP(channels=3)
    model = model.cuda()
    model.load_state_dict(torch.load(opt.modelDir))
    model.eval()

    # process data

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

        # cv2.imwrite(os.path.join(opt.outDir, f), out*255)
        # print("%s PSNR %f SSIM %f" % (f, psnr, ssim))
    psnr_test /= len(os.listdir(opt.rainDir))
    ssim_test /= len(os.listdir(opt.rainDir))

    print("PSNR on test data %f SSIM on test data %f\n" % (psnr_test, ssim_test))
    return ssim_test, psnr_test


if __name__ == "__main__":
    l = len(os.listdir(opt.modelDir))
    dir = opt.modelDir
    epcho=list()
    ssim=list()
    psnr=list()
    for i in range(1, l+1):
        opt.modelDir = os.path.join(dir, 'net'+str(i)+'.pth')
        print(opt.modelDir)
        ssim_test,psnr_test = main()
        epcho.append(i)
        ssim.append(ssim_test)
        psnr.append(psnr_test)
    epcho = np.array(epcho)
    ssim = np.array(ssim)
    psnr = np.array(psnr)
    plt.plot(epcho, ssim, label="SSIM")
    plt.show()
    plt.plot(epcho, psnr, label="PSNR")
    plt.show()