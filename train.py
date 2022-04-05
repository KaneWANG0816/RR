import os
import argparse
import sys
import matplotlib.pyplot as plt
import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from model import DnCNN
from model_Res import DnCNN_Res
from utils import *
from loadDataset import TrainDataset
import time
torch.cuda.empty_cache()
torch.cuda.memory_summary(device=None, abbreviated=False)

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--rainDir", type=str, default="./data/rain100H/train/rain", help='path of rain')
parser.add_argument("--gtDir", type=str, default="./data/rain100H/train/norain", help='path of norain')
parser.add_argument("--maskDir", type=str, default="./data/rain100H/train/masks", help='path of mask')
parser.add_argument("--valRainDir", type=str, default="./data/rain100H/test/rain", help='path of rain for val')
parser.add_argument("--valGtDir", type=str, default="./data/rain100H/test/norain", help='path of norain for val')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument("--batchSize", type=int, default=2, help="Training batch size")
parser.add_argument('--resume', type=int, default=0, help='resume')
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=90, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=89, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--logDir", type=str, default="./logs/rain100H_Res", help='path of log files')
parser.add_argument("--modelDir", type=str, default="./models/rain100H_Res", help='path of models')
parser.add_argument("--trainSize", type=int, default=1800, help='size of training dataset default: float("inf")')
parser.add_argument("--valSize", type=int, default=0, help='size of validation dataset default: float("inf")')
opt = parser.parse_args()


def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cudnn.benchmark = True

    # Load dataset
    print('Loading dataset ...\n')

    dataset_train = TrainDataset(opt.rainDir, opt.gtDir, opt.maskDir,  opt.batchSize*opt.trainSize)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batchSize, shuffle=True)

    print("%d training samples\n" % int(len(dataset_train)))
    # Build model
    # model = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    model = DnCNN_Res(channels=3)
    model.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    model = model.cuda()
    criterion.cuda()
    # Optimizer++++++++
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.logDir)
    # Count time

    if opt.resume:
        model.load_state_dict(torch.load(os.path.join(opt.modelDir, 'net' + str(opt.resume) + '.pth')))

    startTotal = time.time()
    step = opt.resume * opt.trainSize
    for epoch in range(opt.resume, opt.epochs):
        start = time.time()
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.
        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)
        # train
        for i, data in enumerate(loader_train, 0):
            # training step
            input, mask, gt = [x.cuda() for x in data]

            # plt.imshow(input[0].cpu().permute(1, 2, 0))
            # plt.title("x")
            # plt.show()

            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            out_mask = model(input)

            loss = criterion(out_mask, mask) / (input.size()[0] * 2)
            loss.backward()
            optimizer.step()

            if step % 10 == 0:
                # eval
                model.eval()
                out = input - out_mask
                out = torch.clamp(out, 0., 1.)
                psnr_train = batch_PSNR(out, gt, 1.)
                ssim_train = batch_SSIM(out, gt, 1.)
                print("[epoch %d/%d][%d/%d] loss: %.4f PSNR_train: %.4f SSIM_train: %.4f" %
                      (epoch + 1, opt.epochs, i + 1, len(loader_train), loss.item(), psnr_train, ssim_train))
                # Log
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
                writer.add_scalar('SSIM on training data', ssim_train, step)
            if step % 600 == 0:
                rain = utils.make_grid(input.data, normalize=True, scale_each=True)
                derain = utils.make_grid(out.data, normalize=True, scale_each=True)
                norain = utils.make_grid(gt.data, normalize=True, scale_each=True)
                writer.add_image('rain image', rain, epoch)
                writer.add_image('derain image', derain, epoch)
                writer.add_image('norain image', norain, epoch)
            step += 1
        # the end of the epoch
        model.eval()
        psnr_val = 0
        ssim_val = 0
        count = 1
        for f in os.listdir(opt.valRainDir):
            # image
            if count <= opt.valSize:
                rain = cv2.imread(os.path.join(opt.valRainDir, f))
                rain = transform(rain) / 255
                rain = np.transpose(rain, (2, 0, 1))
                rain = np.expand_dims(rain, axis=0)
                rain = torch.Tensor(rain).cuda()

                gt = cv2.imread(os.path.join(opt.valGtDir, f))
                gt = transform(gt) / 255

                mask = model(rain)

                out = torch.clamp(rain - mask, 0., 1.)
                out_tmp = out
                out = out.data.cpu().numpy().astype(np.float32)
                out = np.transpose(out[0, :, :, :], (1, 2, 0))

                psnr = peak_signal_noise_ratio(out, gt, data_range=1)
                ssim = structural_similarity(out, gt, data_range=1, channel_axis=2)
                psnr_val += psnr
                ssim_val += ssim
                count += 1
                if count > opt.valSize:
                    writer.add_image('rain image val', rain[0, :, :, :], epoch)
                    writer.add_image('derain image val', out_tmp[0, :, :, :], epoch)
                    writer.add_image('norain image val', torch.Tensor(gt).permute((2, 0, 1)), epoch)
            else:
                break
        psnr_val /= len(os.listdir(opt.valRainDir))
        ssim_val /= len(os.listdir(opt.valRainDir))
        end = time.time()
        print("\n[epoch %d/%d] PSNR_val: %.4f SSIM_val: %.4f, time cost of epoch: %d s" %
              (epoch + 1, opt.epochs, psnr_val, ssim_val, end - start))
        writer.add_scalar('PSNR on validation dataset', psnr_val, epoch)
        writer.add_scalar('SSIM on validation dataset', ssim_val, epoch)

        # save model
        torch.save(model.state_dict(), os.path.join(opt.modelDir, 'net%d.pth' % (epoch + 1)))
    endTotal = time.time()
    print("\n Total training time cost: %d s" % (endTotal - startTotal))


if __name__ == "__main__":
    main()
