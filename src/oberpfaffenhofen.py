
import os

import pytorch_lightning as pl
import torch
from pytorch_lightning.loggers import TensorBoardLogger
from segmentation_models_pytorch.utils.losses import DiceLoss
from torch import nn
from torch.utils.data import DataLoader
from torchvision import transforms
from models.Polsarnet import PolsarnetModel, RealValuedPolsarnetModel

from models import complexNetwork as cn
from evaluation import (confusion_matrix, kappa_coefficent,
                        mean_producers_accuracy, mean_users_accuracy,
                        overall_accuracy)
from dataloader.oph_dataloader import OPH_Dataset, OPH_Visualization_Dataset


def prepare_oph_normalization(self):
    ts = OPH_Dataset(validation_index=self.validation_index, val=False)
    loader = DataLoader(ts, batch_size=len(ts))
    data,_ = next(iter(loader))

    data_r = data.real
    data_i = data.imag

    mean_r, std_r = data_r.mean(dim=[0,2,3]), data_r.std(dim=[0,2,3])
    mean_i, std_i = data_i.mean(dim=[0,2,3]), data_i.std(dim=[0,2,3])
    
    return mean_r, std_r, mean_i, std_i

def transform_complex_to_real(x):
    return torch.hstack((x.real,x.imag))

        
class Polsarnet(pl.LightningModule):
    def __init__(self, validation_index = 0, learning_rate = 1e-7, weight_decay = 5e-4, version=0, weights_path=None, model='complex'):
        super().__init__()
        self.save_hyperparameters()
        self.lr = learning_rate
        self.weight_decay = weight_decay

        if weights_path:
            self.model = torch.load(weights_path)
        else:
            if model == 'complex':
                self.model = PolsarnetModel(in_channels=6, out_channels=6)
            if model == 'real':
                self.model = RealValuedPolsarnetModel(in_channels=6*2, out_channels=6)
        self.epoch = 0
        self.best_val_loss = 10000
        self.version = version

        weights = [0,1,1,1,1,1]
        weighted_pixelwise_cross_entropy = cn.PixelwiseCrossEntropy(weights)
        pixelwise_cross_entropy = cn.PixelwiseCrossEntropy()
        dice = DiceLoss()

        self.confusion_matrix ={
            'train': torch.zeros((6,6)),
            'val':   torch.zeros((6,6)),
            'test':  torch.zeros((6,6))
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
        self.model.save_weights( f'models/oph/version{self.version}')

    def load_best_version(self):
        self.model = torch.load( f'models/oph/version{self.version}')

    ## Data preperation

    def setup(self, stage):
        # setup z normalization for the datasets
        # imaginary part and real part are normalized independently
        # if self.z_normalize is set to false, skip the normalization
        if self.z_normalize:
            mean_r = torch.Tensor([ 5.8648,  0.2611, -0.0597,  4.6246, -0.0570,  1.0996])
            std_r = torch.Tensor([307.3737, 196.4060,  42.0292, 194.0364,  34.4647,  27.4487])
            mean_i = torch.Tensor([ 0.0000, -0.5121,  0.0306,  0.0000, -0.0599,  0.0000])
            std_i = torch.Tensor([ 1, 75.1370, 38.8940, 1, 33.7835,  1])

            transform_r = transforms.Normalize(mean_r, std_r)
            transform_i = transforms.Normalize(mean_i, std_i)
        else:
            transform_r = None
            transform_i = None

        self.train_dataset = OPH_Dataset(
            validation_index=self.validation_index,
            mode='train',
            real_transform=transform_r,
            imag_transform=transform_i)

        self.val_dataset = OPH_Dataset(
            validation_index=self.validation_index,
            mode='val',
            real_transform=transform_r,
            imag_transform=transform_i)

        self.test_dataset = OPH_Dataset(
            validation_index=self.validation_index,
            mode='test',
            real_transform=transform_r,
            imag_transform=transform_i)

        visualization_dataset = OPH_Visualization_Dataset(
            real_transform=transform_r,
            imag_transform=transform_i)

        self.visualization_dataloader = DataLoader(visualization_dataset, batch_size=106, shuffle=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        
    def val_dataloader(self):        
        return DataLoader(self.val_dataset, batch_size=64, shuffle=False)

    def test_dataloader(self):        
        return DataLoader(self.test_dataset, batch_size=64, shuffle=False)
    
    ## Training logic

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

    ## Evaluation methods

    def log_metrics(self, mode):
        self.log(f'{mode}/overall_accuracy', overall_accuracy(self.confusion_matrix[mode], ignore_index_0=True))
        self.log(f'{mode}/kappa_coefficent', kappa_coefficent(self.confusion_matrix[mode], ignore_index_0=True))
        self.log(f'{mode}/mean_users_accuracy', mean_users_accuracy(self.confusion_matrix[mode], ignore_index_0=True))
        self.log(f'{mode}/mean_producers_accuracy', mean_producers_accuracy(self.confusion_matrix[mode], ignore_index_0=True))

    def save_confusion_matrix(self, prediction, label, mode):
        self.confusion_matrix[mode] = self.confusion_matrix[mode] + confusion_matrix(prediction, label)
    
    def visualize_progress(self):
        image = self.infere_oph()
        tensorboard = self.logger.experiment
        tensorboard.add_image("Prediction", image, global_step = self.epoch, dataformats='HWC')

    def infere_oph(self):
        with torch.no_grad():
            # make predictions
            predictions = []
            for batch in self.visualization_dataloader:
                data, _ = batch
                data = transform_complex_to_real(data).to(self.device)
                prediction = self(data)
                prediction = torch.argmax(prediction, axis=1)
                prediction = torch.hstack([prediction[i] for i in range(106)])
                predictions.append(prediction)
            classes = torch.vstack(predictions).cpu()

            # colorize predicted classes
            image = torch.zeros((6630,1378,3))
            values = [[0,0,0],[0,0,1],[0,0.5,0],[0,1,0],[1,0,0],[1,1,0]]
            for i, val in enumerate(values):
                tmp = torch.zeros((6630,1378,3))
                tmp[:,:] = torch.Tensor(val)
                image[:,:,0] = torch.where(classes == i, tmp[:,:,0], image[:,:,0])
                image[:,:,1] = torch.where(classes == i, tmp[:,:,1], image[:,:,1])
                image[:,:,2] = torch.where(classes == i, tmp[:,:,2], image[:,:,2])

        return image

    ## Optimizer

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        return optimizer

if __name__ == '__main__':
    # train Polsarnet
    for val_index in [0,2,3,4]:
        version = len(os.listdir('logs/oph'))
        logger = TensorBoardLogger(f'logs/', name='oph')
        model = Polsarnet(learning_rate=1e-5, validation_index=val_index, version=version)
        trainer = pl.Trainer(gpus=1, max_epochs=2000, logger=logger)
        trainer.fit(model)
        model.load_best_version()
        trainer.test()

    # train FCN_r6 - the realvalued counterpart
    for val_index in [0,2,3,4]:
        version = len(os.listdir('logs/oph'))
        logger = TensorBoardLogger(f'logs/', name='oph')
        model = Polsarnet(learning_rate=1e-5, validation_index=val_index, version=version, model='real')
        trainer = pl.Trainer(gpus=1, max_epochs=2000, logger=logger)
        trainer.fit(model)
        model.load_best_version()
        trainer.test()
