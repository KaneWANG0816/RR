import argparse
import os
import numpy as np
from matplotlib import pyplot as plt
from RNetP import RNetP
from RNet import RNet
from utils import *

os.environ["CUDA_VISIBLE_DEVICES"] = "0"

parser = argparse.ArgumentParser(description="Derain Test")
parser.add_argument("--model", type=str, default="RNetP", help='Derain model: RNetP or RNet')
opt = parser.parse_args()


def main(mod, dataset):
    # Build model
    if mod == 'RNetP':
        model = RNetP(channels=3)
        print('Loading model\n')
        model = model.cuda()
        if dataset == 'Rain100L':
            model.load_state_dict(torch.load('./model/RNet+Rain100L.pth'))
        else:
            model.load_state_dict(torch.load('./model/RNet+Rain100H.pth'))
    else:
        model = RNet(channels=3)
        print('Loading model\n')
        model = model.cuda()
        if dataset == 'Rain100L':
            model.load_state_dict(torch.load('./model/RNetRain100L.pth'))

        else:
            model.load_state_dict(torch.load('./model/RNetRain100H.pth'))

    if dataset == 'Rain100L':
        rainDir = './samples/Rain100L'
        outDir = './out/Rain100L'
    else:
        rainDir = './samples/Rain100H'
        outDir = './out/Rain100H'

    model.eval()

    # Process
    print('Deraining\n')
    for f in os.listdir(rainDir):
        # image
        print(os.path.join(rainDir, f))
        rain = cv2.imread(os.path.join(rainDir, f)) / 255
        rain = np.transpose(rain, (2, 0, 1))
        rain = np.expand_dims(rain, axis=0)
        rain = torch.Tensor(rain).cuda()

        mask = model(rain)
        out = torch.clamp(rain - mask, 0., 1.)
        out = out.data.cpu().numpy().astype(np.float32)
        out = np.transpose(out[0, :, :, :], (1, 2, 0))

        cv2.imwrite(os.path.join(outDir, f), out * 255)
    print('Derain with ' + mod + ' on ' + dataset + ' is done')


if __name__ == "__main__":
    print(opt.model + '\n')
    main(opt.model, 'Rain100L')
    main(opt.model, 'Rain100H')
