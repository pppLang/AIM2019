import torch
import numpy as np
from torch import nn
from skimage.measure import compare_ssim, compare_psnr


def batch_PSNR(im_true, im_fake, data_range=255):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
    Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W)
    psnr = 10. * torch.log((data_range**2) / err) / np.log(10.)
    return torch.mean(psnr)


def new_ssim(img1, img2, scale=2, data_range=1):
    if isinstance(img1, torch.Tensor):
        img1, img2 = img1.cpu().numpy(), img2.cpu().numpy()
    # img1, img2 = (img1*255).round().astype('uint8'), (img2*255).round().astype('uint8')
    # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCR_CB)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCR_CB)
    if len(img1.shape) == 4:
        img1, img2 = img1[0, 0, :, :], img2[0, 0, :, :]
    if len(img1.shape) == 3:
        img1, img2 = img1[0, :, :], img2[0, :, :]
    img1, img2 = img1[scale:-scale, scale:-scale], img2[scale:-scale, scale:-scale]
    # img1, img2 = img1.astype('float32') / 255., img2.astype('float32') / 255.
    # print('before SSIM, max {}, min {}'.format(img1.max(), img1.min()))
    return compare_ssim(img1, img2, win_size=11, data_range=data_range, gaussian_weights=True)


def new_psnr(img1, img2, scale=2, data_range=1):
    if isinstance(img1, torch.Tensor):
        img1, img2 = img1.data.float().clamp_(0, 1).cpu().numpy(), img2.data.float().clamp_(0, 1).cpu().numpy()
    # img1, img2 = (img1*255).round().astype('uint8'), (img2*255).round().astype('uint8')
    # img1 = cv2.cvtColor(img1, cv2.COLOR_RGB2YCR_CB)
    # img2 = cv2.cvtColor(img2, cv2.COLOR_RGB2YCR_CB)
    if len(img1.shape) == 4:
        img1, img2 = img1[0, 0, :, :], img2[0, 0, :, :]
    if len(img1.shape) == 3:
        img1, img2 = img1[0, :, :], img2[0, :, :]
    img1, img2 = img1[scale:-scale, scale:-scale], img2[scale:-scale, scale:-scale]
    # img1, img2 = img1.astype('float32') / 255., img2.astype('float32') / 255.
    # print('before PSNR, max {}, min {}'.format(img1.max(), img1.min()))
    return compare_psnr(img1, img2, data_range=data_range)


def batch_RMSE_G(im_true, im_fake, data_range=255.):
    N = im_true.size()[0]
    C = im_true.size()[1]
    H = im_true.size()[2]
    W = im_true.size()[3]
    Itrue = im_true.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
    Ifake = im_fake.clamp(0., 1.).mul_(data_range).resize_(N, C * H * W)
    mse = nn.MSELoss(reduce=False)
    err = mse(Itrue, Ifake).sum(dim=1, keepdim=True).div_(C * H * W).sqrt_()
    return torch.mean(err)


def get_parameters(model):
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    return number_parameters


def get_runtime(model):
    start = torch.cuda.Event(enable_timing=True)
    end = torch.cuda.Event(enable_timing=True)
    num = 20
    all_time = 0
    for i in range(num):
        img_L = torch.rand(1, 3, 510, 384).cuda()
        start.record()
        img_E = model(img_L)
        end.record()
        torch.cuda.synchronize()
        time = start.elapsed_time(end)
        print('rand test {}, time {}'.format(i, time))
        all_time += time
    print('{} rand input, avg time {}'.format(num, all_time / (num * 1000)))
    return all_time / (num * 1000)