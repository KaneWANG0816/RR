import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.utils as utils
from torch.autograd import Variable
from torch.backends import cudnn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter
from models import DnCNN
from utils import *
from loadDataset import TrainDataset


os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="DnCNN")
parser.add_argument("--rainDir", type=str, default="./data/train/rain", help='path of rain')
parser.add_argument("--gtDir", type=str, default="./data/train/norain", help='path of norain')
parser.add_argument("--maskDir", type=str, default="./data/train/masks", help='path of mask')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=4)
parser.add_argument("--preprocess", type=bool, default=False, help='run prepare_data or not')
parser.add_argument("--batchSize", type=int, default=3, help="Training batch size")
parser.add_argument("--num_of_layers", type=int, default=17, help="Number of total layers")
parser.add_argument("--epochs", type=int, default=50, help="Number of training epochs")
parser.add_argument("--milestone", type=int, default=30, help="When to decay learning rate; should be less than epochs")
parser.add_argument("--lr", type=float, default=1e-3, help="Initial learning rate")
parser.add_argument("--logDir", type=str, default="./logs", help='path of log files')
parser.add_argument("--modelDir", type=str, default="./models", help='path of models')
opt = parser.parse_args()

def main():
    os.environ["CUDA_VISIBLE_DEVICES"] = "0"
    cudnn.benchmark = True

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = TrainDataset(opt.rainDir, opt.gtDir,opt.maskDir, opt.batchSize * 1000)
    loader_train = DataLoader(dataset=dataset_train, num_workers=0, batch_size=opt.batchSize, shuffle=True)

    print("# of training samples: %d\n" % int(len(dataset_train)))
    # Build model
    model = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    model.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)
    # Move to GPU
    model = model.cuda()
    criterion.cuda()
    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)
    # training
    writer = SummaryWriter(opt.logDir)
    step = 0
    for epoch in range(opt.epochs):
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
            
            model.train()
            model.zero_grad()
            optimizer.zero_grad()

            out_mask = model(input)

            loss = criterion(out_mask, mask) / (input.size()[0]*2)
            loss.backward()
            optimizer.step()
            # results
            model.eval()
            out = input - out_mask
            out = torch.clamp(out, 0., 1.)
            psnr_train = batch_PSNR(out, gt, 1.)
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                (epoch+1, i+1, len(loader_train), loss.item(), psnr_train))
            # if you are using older version of PyTorch, you may need to change loss.item() to loss.data[0]
            if step % 10 == 0:
                # Log the scalar values
                writer.add_scalar('loss', loss.item(), step)
                writer.add_scalar('PSNR on training data', psnr_train, step)
            if step % 500 == 0:
                rain = utils.make_grid(input.data, nrow=8, normalize=True, scale_each=True)
                derain = utils.make_grid(out.data, nrow=8, normalize=True, scale_each=True)
                norain = utils.make_grid(gt.data, nrow=8, normalize=True, scale_each=True)
                writer.add_image('rain image', rain, epoch)
                writer.add_image('derain image', derain, epoch)
                writer.add_image('norain image', norain, epoch)
            step += 1
        # ## the end of each epoch
        # model.eval()
        # # validate
        # psnr_val = 0
        # for k in range(len(dataset_val)):
        #     input_val = torch.unsqueeze(dataset_val[k], 0)
        #     mask = model(input_val)
        #     out = input_val
        #     input_val, imgn_val = Variable(input_val.cuda(), volatile=True), Variable(imgn_val.cuda(), volatile=True)
        #     out_val = torch.clamp(imgn_val-model(imgn_val), 0., 1.)
        #     psnr_val += batch_PSNR(out_val, input_val, 1.)
        # psnr_val /= len(dataset_val)
        # print("\n[epoch %d] PSNR_val: %.4f" % (epoch+1, psnr_val))
        # writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        # out = torch.clamp(input-model(input), 0., 1.)
        # Img = utils.make_grid(input.data, nrow=8, normalize=True, scale_each=True)
        # Imgn = utils.make_grid(input.data, nrow=8, normalize=True, scale_each=True)
        # Irecon = utils.make_grid(out.data, nrow=8, normalize=True, scale_each=True)
        # writer.add_image('clean image', Img, epoch)
        # writer.add_image('noisy image', Imgn, epoch)
        # writer.add_image('reconstructed image', Irecon, epoch)
        # save model
        torch.save(model.state_dict(), os.path.join(opt.modelDir, 'net.pth'))


if __name__ == "__main__":
    main()
