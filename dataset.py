import random
import torch
import torch.utils.data as udata
import h5py
import numpy as np
from evalution import new_psnr, new_ssim, batch_RMSE_G


class MyDataset(udata.Dataset): 
    def __init__(self, mode='train', scaling_factor=2, input_large=False, selected_num=None):
        self.mode = mode
        self.scaling_factor = scaling_factor
        self.input_large = input_large
        print('start loading')
        if input_large:
            print('input large')
            self.h5f = h5py.File('DIV2K/DIV2K_train_LR_bicubic/cv2_data_{}_x{}L.h5'.format(selected_num, self.scaling_factor), 'r')
        else:
            print('not input large')
            self.h5f = h5py.File('DIV2K/DIV2K_train_LR_bicubic/cv2_traindata_{}_x{}.h5'.format(selected_num, self.scaling_factor), 'r')
        print('has load dataset')
        self.keys = list(range(len(self.h5f.keys()) // 2))
        print('has load keys')
        random.shuffle(self.keys)
        print('has shuffle keys')
        print('total {} samples '.format(len(self.keys)))

    def __len__(self):
        return len(self.keys)

    def close(self):
        self.h5f.close()

    def __getitem__(self, index):
        hr, lr = self.h5f['{}_hr'.format(self.keys[index])], self.h5f['{}_lr'.format(self.keys[index])]
        hr, lr = torch.Tensor(np.array(hr)), torch.Tensor(np.array(lr))
        return hr, lr


def test_bicubic(testDataset):
    num = len(testDataset)
    psnr_sum, rmse_sum, ssim_sum = 0, 0, 0
    for i, (hr, lr) in enumerate(testDataset):
        hr, hr_fake = hr.unsqueeze(0).unsqueeze(0).cuda(), lr.unsqueeze(0).unsqueeze(0).cuda()
        psnr = new_psnr(hr, hr_fake, scale=2, data_range=1)
        rmse = batch_RMSE_G(hr, hr_fake, data_range=1)
        ssim = new_ssim(hr, hr_fake, scale=2, data_range=1)
        print('img {}, psnr {}, rmse {}, ssim {}'.format(i, psnr, rmse, ssim))
        psnr_sum += psnr
        rmse_sum += rmse
        ssim_sum += ssim

    psnr_sum, rmse_sum, ssim_sum = psnr_sum / num, rmse_sum / num, ssim_sum / num
    print('epoch {}, {} imgs , avg psnr {}, avg rmse {}, avg ssim {}'.format(0, num, psnr_sum, rmse_sum, ssim_sum))


if __name__ == "__main__":
    import os
    import cv2
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = "7"

    scaling_factor = 2
    input_large = True
    selected_num = None
    testDataset = MyTestDataset('DIV2K_valid_HR', scaling_factor=scaling_factor, input_large=input_large)
    test_bicubic(testDataset)
    exit()
    # random.shuffle(testDataset.keys)
    print('total test sample num {}'.format(len(testDataset)))
    testDataset.close()
