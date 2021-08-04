from functools import lru_cache

import matplotlib.pyplot as plt
import numpy as np
import numpy.random as rng
import torch
from PIL import Image
from torch.utils import data
from torch.utils.data import Dataset

from dataloader.data_helpers import box_filter, to_coherency, to_pauli


@lru_cache
def load_poland_eval_files():
    data = np.load('./data/Poland/val&test/sentinel1.npy')

    # create the labels of the label images
    labels = np.ones((8496, 14121))
    labels *= -1
    for index, name in enumerate(['field', 'forest', 'street', 'urban', 'water']):
        im = np.asarray(Image.open(f'data/Poland/val&test/labels/{name}.png'))
        labels = np.where(im, index, labels)

    # reorder the axes
    data = np.transpose(data, (2, 0, 1))

    # just ignore the mask for now
    return data, labels


@lru_cache
def load_poland_train_files():
    data = np.load('./data/Poland/train/sentinel1.npy')

    # create the labels of the label images
    labels = np.ones((8713, 13980))
    labels *= -1
    for index, name in enumerate(['field', 'forest', 'street', 'urban', 'water']):
        im = np.asarray(Image.open(
            f'data/Poland/train/labels/{name}.png'))[:, :, 0]
        labels = np.where(im, index, labels)

    # reorder the axes
    data = np.transpose(data, (2, 0, 1))

    return data, labels


def sample_patches(x, y):
    _, ax1, ax2 = x.shape
    patches_x = []
    patches_y = []
    for i in range(ax1//13):
        i *= 13
        for j in range(ax2//13):
            j*=13
            patch_y = y[i:i+13, j:j+13]
            if (patch_y == -1).all():
                continue
            patch_x = x[:, i:i+13, j:j+13]
            patches_x.append(patch_x)
            patches_y.append(patch_y)

    return np.stack(patches_x), np.stack(patches_y)


class Poland_Dataset(Dataset):

    def __init__(self, mode, real_transform = None, imag_transform = None):
        
        if mode == 'train':
            self.x = np.load('data/Poland/train/small_train_patches_x.npy')
            self.y = np.load('data/Poland/train/small_train_patches_y.npy')
        if mode == 'val':
            self.x = np.load('data/Poland/val&test/small_val_patches_x.npy')
            self.y = np.load('data/Poland/val&test/small_val_patches_y.npy')
        if mode == 'test':
            self.x = np.load('data/Poland/val&test/small_test_patches_x.npy')
            self.y = np.load('data/Poland/val&test/small_test_patches_y.npy')

        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)
        self.y[self.y == -1] = 5
        a = self.x.real
        
        if real_transform and imag_transform:
            r = real_transform(self.x.real)
            i = imag_transform(self.x.imag) * 1j
            self.x = r + i
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class Poland_Visualization_Dataset(Dataset):

    def __init__(self, real_transform = None, imag_transform = None):
        data_x, eval_y = load_poland_eval_files()
    
        data_x = np.stack([data_x[0] + data_x[1] * 1j,
                           data_x[2] + data_x[3] * 1j])

        print(data_x.shape)

        eval_x = box_filter(to_coherency(data_x))


        val_x = eval_x[:,:,:7090]
        val_y = eval_y[:,:7090]
        val_patches_x, val_patches_y = sample_patches(val_x, val_y)

        self.x = torch.from_numpy(self.x)
        self.y = torch.from_numpy(self.y)
        self.y[self.y == -1] = 5
        a = self.x.real
        
        if real_transform and imag_transform:
            r = real_transform(self.x.real)
            i = imag_transform(self.x.imag) * 1j
            self.x = r + i
    
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

if __name__ == '__main__':
    a = Poland_Dataset('test')
    # How to extract the patches: 

    # data_x, train_y = load_poland_train_files()
    # data_x = np.stack([data_x[0] + data_x[1] * 1j,
    #                    data_x[2] + data_x[3] * 1j])
    # train_x = box_filter(to_coherency(data_x))
    # train_patches_x, train_patches_y = sample_patches(train_x, train_y)
    # np.save('train_patches_x.npy', train_patches_x)
    # np.save('train_patches_y.npy', train_patches_y)
    # print(len(train_patches_x))

    # data_x, eval_y = load_poland_eval_files()
    
    # data_x = np.stack([data_x[0] + data_x[1] * 1j,
    #                    data_x[2] + data_x[3] * 1j])

    # print(data_x.shape)

    # eval_x = box_filter(to_coherency(data_x))


    # val_x = eval_x[:,:,:7060]
    # val_y = eval_y[:,:7090]
    # val_patches_x, val_patches_y = sample_patches(val_x, val_y)
    # np.save('small_val_patches_y.npy', val_patches_y)
    # np.save('small_val_patches_x.npy', val_patches_x)
    # print(len(val_patches_x))

    # test_x = eval_x[:,:,7060:]
    # test_y = eval_y[:,7060:]
    # test_patches_x, test_patches_y = sample_patches(test_x, test_y)
    # np.save('small_test_patches_y.npy', test_y)
    # np.save('small_test_patches_x.npy', test_x)
    # print(len(test_patches_x))
