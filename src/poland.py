import os
from numpy.lib.npyio import load

import pytorch_lightning as pl
import torch
import numpy as np
from pytorch_lightning.loggers import TensorBoardLogger
from torch.utils.data import DataLoader
from torchvision import transforms

import models.complexNetwork as cn
from evaluation import (confusion_matrix, kappa_coefficent,
                        mean_producers_accuracy, mean_users_accuracy,
                        overall_accuracy)
from dataloader.poland_dataloader import Poland_Dataset, load_poland_eval_files
from models.Polsarnet import PolsarnetModel
from dataloader.data_helpers import box_filter, to_coherency

def transform_complex_to_real(x):
    return torch.hstack((x.real,x.imag))

def prepare_poland_normalization():
    ts = Poland_Dataset(mode='train')
    loader = DataLoader(ts, batch_size=len(ts))
    data,_ = next(iter(loader))

    data_r = data.real
    data_i = data.imag

    mean_r, std_r = data_r.mean(dim=[0,2,3]), data_r.std(dim=[0,2,3])
    mean_i, std_i = data_i.mean(dim=[0,2,3]), data_i.std(dim=[0,2,3])
    
    return mean_r, std_r, mean_i, std_i


class Polsarnet(pl.LightningModule):

    def __init__(self, validation_index=0, learning_rate=1e-7, weight_decay=5e-4, version=0, weights_path=None):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.weight_decay = weight_decay

        if weights_path:
            self.model = torch.load(weights_path)
        else:
            self.model = PolsarnetModel(in_channels=3, out_channels=6)
        self.epoch = 0
        self.best_val_loss = 10000
        self.version = version

        weighted_pixelwise_cross_entropy = cn.PixelwiseCrossEntropy(one_hot_encoded=False)

        self.confusion_matrix = {
            'train': torch.zeros((5, 5)),
            'val':   torch.zeros((5, 5)),
            'test':  torch.zeros((5, 5))
        }

        self.losses = {
            'val': [],
        }

        self.validation_index = validation_index
        self.z_normalize = True
        self.loss_function = weighted_pixelwise_cross_entropy
        self.visualize_every_n_epochs = 10

    def forward(self, x):
        return self.model(x)

    def save_weights(self):
        self.model.save_weights(f'models/poland/version{self.version}')

    def load_best_version(self):
        self.model = torch.load(f'models/poland/version{self.version}')

    # Data preperation

    def setup(self, stage):
        if self.z_normalize:
            mean_r = torch.Tensor([ 3.2249e+09, -1.9232e+10,  2.4078e+11])
            std_r = torch.Tensor([3.9890e+11, 3.6551e+12, 3.4768e+13])
            mean_i = torch.Tensor([ 0.0000e+00, -2.8493e+09,  0.0000e+00])
            std_i = torch.Tensor([1e-15, 3.1006e+11, 1e-15])
            transform_r = transforms.Normalize(mean_r, std_r)
            transform_i = transforms.Normalize(mean_i, std_i)
        else:
            transform_r = None
            transform_i = None

        self.train_dataset = Poland_Dataset(mode='train', real_transform=transform_r, imag_transform=transform_i)
        self.val_dataset = Poland_Dataset(mode='val', real_transform=transform_r, imag_transform=transform_i)
        self.test_dataset = Poland_Dataset(mode='test', real_transform=transform_r, imag_transform=transform_i)

        visualization_data, _ = load_poland_eval_files()
        visualization_data = visualization_data[:,:8489,:7059]
        visualization_data = np.stack([visualization_data[0] + visualization_data[1] * 1j,
                    visualization_data[2] + visualization_data[3] * 1j])
        visualization_data = box_filter(to_coherency(visualization_data))
        visualization_data = torch.from_numpy(visualization_data)
        visualization_data = transform_r(visualization_data.real) + transform_i(visualization_data.imag) * 1j
        visualization_patches = []
        for i in range(8489//13):
            i *= 13
            patch_row = []
            for j in range(7059//13):
                j *= 13
                patch = visualization_data[:,i:i+13,j:j+13]
                patch_row.append(patch)
            visualization_patches.append(torch.stack(patch_row))

        self.visualization_patches = torch.stack(visualization_patches)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True)

    def val_dataloader(self):
        return DataLoader(self.val_dataset, batch_size=64, shuffle=False)

    def test_dataloader(self):
        return DataLoader(self.test_dataset, batch_size=64, shuffle=False)

    # Training logic

    def step(self, batch):
        data, label = batch
        data = transform_complex_to_real(data)

        data, label = data.double(), label.double()
        prediction = self(data)
        loss = self.loss_function(prediction, label)
        return data, label, prediction, loss

    def training_step(self, batch, batch_idx):
        data, label, prediction, loss = self.step(batch)
        self.save_confusion_matrix(prediction, label, 'train')
        self.log('train/loss', loss)
        return loss

    def on_train_epoch_end(self):
        self.log_metrics('train')
        self.confusion_matrix['train'] *= 0
        return super().on_train_epoch_end()

    def validation_step(self, batch, batch_idx):
        data, label, prediction, loss = self.step(batch)
        self.save_confusion_matrix(prediction, label, 'val')
        self.losses['val'].append(loss)
        self.log('val/loss', loss)
        return loss

    def on_validation_epoch_end(self):
        self.log_metrics('val')
        self.confusion_matrix['val'] *= 0

        avg_loss = sum(self.losses['val']) / len(self.losses['val'])
        if avg_loss < self.best_val_loss:
            self.best_val_loss = avg_loss
            self.save_weights()
        self.losses['val'] = []

        self.epoch += 1
        if self.epoch % self.visualize_every_n_epochs == 1:
            #pass
            self.visualize_progress()

        return super().on_validation_epoch_end()

    def test_step(self, batch, batch_idx):
        data, label, prediction, loss = self.step(batch)
        self.save_confusion_matrix(prediction, label, 'test')
        self.log('test/loss', loss)
        return loss

    def on_test_epoch_end(self):
        self.log_metrics('test')
        self.confusion_matrix['test'] *= 0

        return super().on_test_epoch_end()

    def log_metrics(self, mode):
        self.log(f'{mode}/overall_accuracy',
                 overall_accuracy(self.confusion_matrix[mode], ignore_index_0=True))
        self.log(f'{mode}/kappa_coefficent',
                 kappa_coefficent(self.confusion_matrix[mode], ignore_index_0=True))
        self.log(f'{mode}/mean_users_accuracy',
                 mean_users_accuracy(self.confusion_matrix[mode], ignore_index_0=True))
        self.log(f'{mode}/mean_producers_accuracy',
                 mean_producers_accuracy(self.confusion_matrix[mode], ignore_index_0=True))

    def save_confusion_matrix(self, prediction, label, mode):
        self.confusion_matrix[mode] = self.confusion_matrix[mode] + \
            confusion_matrix(prediction, label, one_hot_encoded=False)

    def visualize_progress(self):
        image = self.predict_poland()
        tensorboard = self.logger.experiment
        tensorboard.add_image("Prediction", image, global_step = self.epoch, dataformats='HWC')

    def predict_poland(self):
        with torch.no_grad():
            predictions = torch.zeros(6, 8489, 7059)
            rows, *_ = self.visualization_patches.shape
            for i in range(rows):
                row = self(transform_complex_to_real(self.visualization_patches[i]).to('cuda'))
                predictions[:,i*13:i*13+13] = row.view(6,13,543*13).to('cpu')
        predictions = torch.argmax(predictions, dim=0)
        predictions = predictions.view(653*13, 13*543)

        # colorize predicted classes
        image = torch.zeros(8489, 7059, 3)
        colors = torch.Tensor([[0.4, 0.2, 0.0], [0.2, 0.4, 0.0], [0.7, 0.7, 0.8], [1.0, 0.4, 0.0], [0.0, 0.0, 0.8], [1.0, 1.0, 1.0]])
        for i, color in enumerate(colors):
            image[predictions==i] = color
        return image

    # Optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(
            self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer


if __name__ == '__main__':
    # print(prepare_poland_normalization())

    version = len(os.listdir('logs/oph'))
    logger = TensorBoardLogger(f'logs/', name='poland')
    model = Polsarnet(learning_rate=1e-5, version=version)
    trainer = pl.Trainer(gpus=1, max_epochs=5000, logger=logger)
    trainer.fit(model)
    model.load_best_version()
    trainer.test()
