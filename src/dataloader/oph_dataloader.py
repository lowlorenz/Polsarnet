import numpy as np
import numpy.random as rng

import cv2

from PIL import Image
from torch.utils.data import Dataset
import torch

from functools import lru_cache

from dataloader.data_helpers import to_coherency, to_pauli, box_filter

#@lru_cache
def load_oph():
    data  = np.load('./data/oph/oph_data.npy')
    x = data[:,:,:,0] + data[:,:,:,1] * 1j
    x = np.flip(x, axis=1)

    image = Image.open('./data/oph/label_inter.png')
    y = np.asarray(image)
    return x, y

def oph_to_categorical(y):      
    '''
    transforms the 6 rgb values to a one hot encoding 
    '''
    y_categorical = np.zeros((6640,1390,6))
    # schwarz, blau, dunkelgrün, hellgrün, rot, gelb
    values = [[0,0,0],[0,0,255],[0,128,0],[0,255,0],[255,0,0],[255,255,0]]
    for i, val in enumerate(values):
        ind = np.where(np.all(y == val, axis=-1))
        y_categorical[ind[0], ind[1], i] = 1
    
    y_categorical = np.transpose(y_categorical, (2, 0, 1))
    return y_categorical

def oph_split_into_strips(x,y):
    '''
    splits the whole dataset in 5 strips on the x axis, so that cross validation can be used
    '''
    x_splits = [ x[:,:,:278], x[:,:,278*1:278*2], x[:,:,278*2:278*3], x[:,:,278*3:278*4], x[:,:,278*4:278*5] ]
    y_splits = [ y[:,:,:278], y[:,:,278*1:278*2], y[:,:,278*2:278*3], y[:,:,278*3:278*4], y[:,:,278*4:278*5] ]
    return x_splits, y_splits

def oph_extract_train_patches(x_splits, y_splits, validation_patch=0, test_patch=1, num_samples=30000):
    train_x_splits = [x for i,x in enumerate(x_splits) if i!=validation_patch and i!=test_patch]
    train_y_splits = [y for i,y in enumerate(y_splits) if i!=validation_patch and i!=test_patch]
    
    train_x_patches = np.zeros((6,13,13, num_samples),dtype=np.complex)
    train_y_patches = np.zeros((6,13,13, num_samples),dtype=np.float)

    indices = np.stack([rng.randint(0,6627,(num_samples)), rng.randint(0, 265, (num_samples))], axis=-1)
    for i in range(num_samples//3): 
        for j in range(3):
            indx0, indx1 = indices[i]
            train_x_patches[:,:,:,i] = train_x_splits[j][:,indx0:indx0+13, indx1:indx1+13]
            train_y_patches[:,:,:,i] = train_y_splits[j][:,indx0:indx0+13, indx1:indx1+13]
            i += num_samples//3
    return train_x_patches, train_y_patches

def oph_extract_validation_patches(x_splits, y_splits, validation_patch=0, num_samples=15000):
    val_x_split = x_splits[validation_patch]
    val_y_split = y_splits[validation_patch]

    val_x_patches = np.zeros((6,13,13,num_samples),dtype=np.complex)
    val_y_patches = np.zeros((6,13,13,num_samples),dtype=np.float)

    indices = np.stack([rng.randint(0,6627,(num_samples)), rng.randint(0, 265,(num_samples))], axis=-1)
    for i in range(num_samples): 
        indx0, indx1 = indices[i]
        val_x_patches[:,:,:,i] = val_x_split[:,indx0:indx0+13, indx1:indx1+13]
        val_y_patches[:,:,:,i] = val_y_split[:,indx0:indx0+13, indx1:indx1+13]
    
    return val_x_patches, val_y_patches

def oph_extract_patches(x_splits, y_splits, validation_patch=0, test_patch=1, mode='val'):
    '''
    extracts random 13x13 patches from the splits.
    in the paper 12k training patches were used for training - therefore we extract 3k patches from 4 of the splits for training and 3k from the remaining validation patch
    '''
    if mode == 'val':
        return oph_extract_validation_patches(x_splits, y_splits, validation_patch)
    if mode == 'test':
        return oph_extract_validation_patches(x_splits, y_splits, test_patch)
    if mode == 'train':
        return oph_extract_train_patches(x_splits, y_splits, validation_patch, test_patch)

def generate_splits():
    x,y = load_oph()
    coherency = box_filter(to_coherency(to_pauli(x)))
    y_categorical = oph_to_categorical(y)
    return oph_split_into_strips(coherency, y_categorical)

def extract_visualization_patches(x,y):
    
    long_ax_range = 6640//13
    short_ax_range = 1390//13
    dim = long_ax_range * short_ax_range

    x_patches = np.zeros((dim,6,13,13,),dtype=np.complex)
    y_patches = np.zeros((dim,6,13,13),dtype=np.float)
    for i in range(long_ax_range):
        for j in range(short_ax_range):
            x_patches[i*short_ax_range + j] = x[:, i*13:(i+1)*13, j*13:(j+1)*13]
            y_patches[i*short_ax_range + j] = y[:, i*13:(i+1)*13, j*13:(j+1)*13]
    return x_patches, y_patches

class OPH_Dataset(Dataset):

    x_splits, y_splits = generate_splits()

    def __init__(self, mode, validation_index = 0, test_index=1, real_transform = None, imag_transform = None):
        x_patches, y_patches = oph_extract_patches(self.x_splits, self.y_splits, validation_patch=validation_index, test_patch=test_index, mode=mode)
        self.x = torch.from_numpy(x_patches).permute(3,0,1,2)
        self.y = torch.from_numpy(y_patches).permute(3,0,1,2)
        if real_transform  and imag_transform:
            self.x = real_transform(self.x.real) + imag_transform(self.x.imag) * 1j

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]

class OPH_Visualization_Dataset(Dataset):
    
    def __init__(self, real_transform = None, imag_transform = None):
        x,y = load_oph()
        coherency = box_filter(to_coherency(to_pauli(x)))
        y_categorical = oph_to_categorical(y)
        x_patches, y_patches = extract_visualization_patches(coherency, y_categorical)
        self.x = torch.from_numpy(x_patches)
        self.y = torch.from_numpy(y_patches)
        if real_transform  and imag_transform:
            self.x = real_transform(self.x.real) + imag_transform(self.x.imag) * 1j
      
    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y[idx]