import os
import torch
from torch import nn
import math
import cv2
import numpy as np
import glob
from utils import utils_image as util
from evalution import new_ssim, new_psnr, batch_PSNR


def train_epoch(model, optimizer, train_loader, criterion, epoch, writer=None):
    num = len(train_loader)
    model.train()
    for i, (hr, lr) in enumerate(train_loader, 1):
        model.zero_grad()
        optimizer.zero_grad()
        hr, lr = hr.cuda(), lr.cuda()
        hr_fake = model.forward(lr)
        loss = criterion(hr_fake, hr)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            with torch.no_grad():
                psnr = batch_PSNR(hr, hr_fake)
                print('epoch {}, [{}/{}]:, loss {}, psnr {}'.format(epoch, i, num, loss, psnr.item()))
                if writer is not None:
                    step = epoch*num + i
                    writer.add_scalar('loss', loss.item(), step)
                    writer.add_scalar('psnr', psnr.item(), step)


def train_epoch_Focal(model, optimizer, train_loader, criterion, criterion_focal, epoch, writer=None):
    num = len(train_loader)
    model.train()
    for i, (hr, lr) in enumerate(train_loader, 1):
        model.zero_grad()
        optimizer.zero_grad()
        hr, lr = hr.cuda(), lr.cuda()
        hr_fake = model.forward(lr)
        loss = criterion(hr_fake, hr) + criterion_focal(hr, hr_fake)
        loss.backward()
        optimizer.step()
        if i % 20 == 0:
            with torch.no_grad():
                psnr = new_psnr(hr, hr_fake)
                print('epoch {}, [{}/{}]:, loss {}, psnr {}'.format(epoch, i, num, loss, psnr))
                if writer is not None:
                    step = epoch*num + i
                    writer.add_scalar('loss', loss.item(), step)
                    writer.add_scalar('psnr', psnr, step)

def test(model, epoch, writer=None, dataset_name='my_val'):
    model.eval()
    L_folder = 'DIV2K/DIV2K_valid_LR_bicubic'
    lr_paths = glob.glob(os.path.join(L_folder, 'X4', '*.png'))
    num = len(lr_paths)
    psnr_sum, ssim_sum = 0, 0
    with torch.no_grad():
        for i, lr_path in enumerate(lr_paths):
            hr_path = lr_path.replace('LR_bicubic', 'HR').replace('X4/', '').replace('x4', '')

            lr = util.imread_uint(lr_path, n_channels=3)
            lr = util.uint2tensor4(lr)
            lr = lr.cuda()
            hr = util.imread_uint(hr_path, n_channels=3)
            hr = util.uint2tensor4(hr)
            hr = hr.cuda()
            
            hr_fake = model.forward(lr)
            hr_fake = hr_fake.clamp(0, 1)
            psnr = new_psnr(hr, hr_fake)
            ssim = new_ssim(hr, hr_fake)
            print('epoch {}, img {}, psnr {}, ssim {}'.format(epoch, i, psnr, ssim))
            psnr_sum += psnr
            ssim_sum += ssim
    psnr_sum, ssim_sum = psnr_sum / num, ssim_sum / num
    print('epoch {}, {} {} imgs , avg psnr {}, avg ssim {}'.format(epoch, dataset_name, num, psnr_sum, ssim_sum))
    if writer is not None:
        writer.add_scalar('psnr_test_{}'.format(dataset_name), psnr_sum, epoch)
        writer.add_scalar('ssim_test_{}'.format(dataset_name), ssim_sum, epoch)
