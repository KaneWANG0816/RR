import os
import os.path
import numpy as np
import torch
import cv2
import torch.utils.data as udata


class TrainDataset(udata.Dataset):
    def __init__(self, rainDir, gtDir, maskDir, length):
        super().__init__()
        self.rainDir = os.path.join(rainDir)
        self.gtDir = os.path.join(gtDir)
        self.rainDir = os.path.join(rainDir)
        self.maskDir = os.path.join(maskDir)
        self.img_files = os.listdir(self.rainDir)
        self.file_num = len(self.img_files)
        self.sample_num = length

    def __len__(self):
        return self.sample_num

    def __getitem__(self, idx):
        file_name = self.img_files[idx % self.file_num]
        img_file = os.path.join(self.rainDir, file_name)
        O = cv2.imread(img_file)
        O = O.astype(np.float32) / 255
        O = np.transpose(O, (2, 0, 1))

        gt_file = os.path.join(self.gtDir, file_name)
        B = cv2.imread(gt_file)
        B = B.astype(np.float32) / 255
        B = np.transpose(B, (2, 0, 1))

        mask_file = os.path.join(self.maskDir, file_name)
        M = cv2.imread(mask_file)
        M = M.astype(np.float32) / 255
        M = np.transpose(M, (2, 0, 1))
        return torch.Tensor(O), torch.Tensor(M), torch.Tensor(B)

