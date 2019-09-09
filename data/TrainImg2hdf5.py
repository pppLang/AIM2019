import os
import sys
import glob
import random
import cv2
import h5py
import pandas as pd
from scipy.io import loadmat
import numpy as np

sys.path.append('../')
import utils.utils_image as util


def Im2Patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw - win + 0 + 1:stride, 0:endh - win + 0 + 1:stride]
    TotalPatNum = patch.shape[1] * patch.shape[2]
    Y = np.zeros([endc, win * win, TotalPatNum], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw - win + i + 1:stride, j:endh - win + j + 1:stride]
            Y[:, k, :] = np.array(patch[:]).reshape(endc, TotalPatNum)
            k = k + 1
    return Y.reshape([endc, win, win, TotalPatNum])


def data_augment(patch, mode=0):
    if mode == 0:
        return patch
    if mode == 1:
        return patch[:, ::-1, :]
    elif mode == 2:
        return patch[:, :, ::-1]
    elif mode == 3:
        return np.rot90(patch, k=1, axes=(1, 2))
    elif mode == 4:
        return np.rot90(patch, k=2, axes=(1, 2))
    elif mode == 5:
        return np.rot90(patch, k=3, axes=(1, 2))


def data_augmentation(patch, index):
    augment_modes = 6
    # augmented_patches = np.zeros([patch.shape[0], patch.shape[1], patch.shape[2], augment_modes])
    augmented_patches = np.zeros([patch.shape[0], patch.shape[1], patch.shape[2], 2])
    # for i in range(augment_modes):
    #     augmented_patches[:, :, :, i] = data_augment(patch, i)
    augmented_patches[:, :, :, 0] = data_augment(patch, 0)
    augmented_patches[:, :, :, 1] = data_augment(patch, index)
    return augmented_patches


def data_augmentation2(patch):
    augment_modes = 6
    augmented_patches = np.zeros([patch.shape[0], patch.shape[1], patch.shape[2], augment_modes])
    for i in range(augment_modes):
        augmented_patches[:, :, :, i] = data_augment(patch, i)
    return augmented_patches


def mat2hdf5(root_lr_path, hr_path, select_num=None, hr_patch_szie=96, hr_stride=96, scale=2):
    print(root_lr_path)
    lr_path = os.path.join(root_lr_path, 'X{}/'.format(scale))
    lr_files_list = glob.glob(os.path.join(lr_path, '*.png'))
    print('total {} imgs !!!'.format(len(lr_files_list)))
    print('mat file list demo : {}'.format(lr_files_list[0]))
    if select_num is None:
        select_num = len(lr_files_list)
    h5_file_path = os.path.join(root_lr_path, 'cv2_traindata_{}_x{}.h5'.format(select_num, scale))
    print('save h5 file path : {}'.format(h5_file_path))
    h5f = h5py.File(h5_file_path, mode='w')
    if not os.path.exists(os.path.join(root_lr_path, 'cv2_{}_filelist_3c.csv'.format(select_num))):
        print('generate new {} filelist'.format(select_num))
        random.shuffle(lr_files_list)
        lr_files_list = lr_files_list[:select_num]
        lr_files_list.sort()
        file_list = np.array(lr_files_list)
        excel = pd.DataFrame(data=file_list, columns=['filename'])
        excel.to_csv(os.path.join(root_lr_path, 'cv2_{}_filelist_3c.csv'.format(select_num)))
    else:
        print('read {} filelist'.format(select_num))
        csv_data = pd.read_csv(os.path.join(root_lr_path, 'cv2_{}_filelist_3c.csv'.format(select_num)))
        csv_data = np.array(csv_data)
        print(csv_data.shape)
        lr_files_list = list(csv_data[:, 1])
    print('total generate {} img files\n\n'.format(len(lr_files_list)))
    index = 0
    for i, lr_file_path in enumerate(lr_files_list):
        hr_file_path = lr_file_path.replace(lr_path, hr_path).replace('x{}'.format(scale), '')
        print(lr_file_path, hr_file_path)
        lr = cv2.imread(lr_file_path)
        lr = cv2.cvtColor(lr, cv2.COLOR_BGR2RGB)
        lr = np.array(lr).astype('float32') / 255
        lr = np.transpose(lr, [2, 0, 1])
        hr = cv2.imread(hr_file_path)
        hr = cv2.cvtColor(hr, cv2.COLOR_BGR2RGB)
        hr = np.array(hr).astype('float32') / 255
        hr = np.transpose(hr, [2, 0, 1])
        if len(hr.shape) == 2 or hr.shape[0] != 3:
            print('spectral shape error !!!! {}, {}'.format(hr.shape, lr.shape))
            continue
        if hr.shape[1] != lr.shape[1] * scale or hr.shape[2] != lr.shape[2] * scale:
            print('spatial shape error !!!! {}, {}'.format(hr.shape, lr.shape))
            continue
        print(hr.shape, hr.max(), hr.min(), hr.dtype)
        print(lr.shape, lr.max(), lr.min(), lr.dtype)
        print('now total {} sample'.format(index))
        hr_patches = Im2Patch(hr, win=hr_patch_szie, stride=hr_stride)
        lr_patches = Im2Patch(lr, win=int(hr_patch_szie / scale), stride=int(hr_stride / scale))
        print('before augumented, total {} patches'.format(hr_patches.shape[3]))
        for j in range(hr_patches.shape[3]):
            hr_patch = hr_patches[:, :, :, j]
            lr_patch = lr_patches[:, :, :, j]
            j = random.randint(1, 5)
            augmented_hr_patches = data_augmentation(hr_patch, j)
            augmented_lr_patches = data_augmentation(lr_patch, j)
            for k in range(augmented_hr_patches.shape[-1]):
                h5f.create_dataset('{}_hr'.format(index), data=augmented_hr_patches[:, :, :, k])
                h5f.create_dataset('{}_lr'.format(index), data=augmented_lr_patches[:, :, :, k])
                index += 1
                # exit()
                # break
        print('each patch, augmentated {} pathces'.format(augmented_hr_patches.shape[3]))
        print('img {}, generate toal {}'.format(i, augmented_hr_patches.shape[3] * hr_patches.shape[3]))
        print('\n')

    h5f.close()

    print('finally total {} samples '.format(index))


if __name__ == "__main__":
    root_lr_path = 'DIV2K/DIV2K_train_LR_bicubic/'
    hr_path = 'DIV2K/DIV2K_train_HR/'
    scales = [4]
    select_num = None
    for scale in scales:
        mat2hdf5(root_lr_path, hr_path, select_num=select_num, scale=scale)
