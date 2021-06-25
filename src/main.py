
import pytorch_lightning as pl
import torch
from torch import nn
import complexNetwork as cn
from oph_dataloader import OPH_Dataset, OPH_VisualizationDataset
from torch.utils.data import DataLoader
from torchvision import transforms
from segmentation_models_pytorch.utils.losses import DiceLoss

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

class AirsarModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.relu = cn.ComplexRelu()
        self.K1_conv = cn.ComplexConv2d(6, 16, 3, dilation=1, stride=1, padding=1)
        self.K2_conv = cn.ComplexConv2d(16, 32, 3, dilation=1, stride=1, padding=1)
        self.K3_conv = cn.ComplexConv2d(32, 32, 3, dilation=1, stride=1, padding=1)
        self.K4_conv = cn.ComplexConv2d(32, 32, 3, dilation=1, stride=1, padding=1)
        self.DK2_conv = cn.ComplexConv2d(32, 32, 3, dilation=2, stride=1, padding=2)
        self.DK3_conv = cn.ComplexConv2d(32, 32, 3, dilation=3, stride=1, padding=3)
        self.Class_conv = cn.ComplexConv2d(32, 6, 1, dilation=1, stride=1, padding=0)
        self.Class_softmax = cn.ComplexSoftmax()

    def forward(self, x):        
        x = self.relu(self.K1_conv(x.float()))
        x = self.relu(self.K2_conv(x))
        x = self.relu(self.K3_conv(x))
        x = self.relu(self.K4_conv(x))
        x = self.relu(self.DK2_conv(x))
        x = self.relu(self.DK3_conv(x))
        y = self.Class_softmax(self.Class_conv(x))
        return y


class Polsarnet(pl.LightningModule):
    def __init__(self, validation_index = 0):
        super().__init__()

        self.model = AirsarModel()
        
        self.validation_index = validation_index
        self.epoch = 0
        
        #weights = [0.125,1,0.25,1,0.5,0.114]
        weights = [0,1,1,1,1,1]
        weighted_pixelwise_cross_entropy = cn.PixelwiseCrossEntropy(weights)
        pixelwise_cross_entropy = cn.PixelwiseCrossEntropy()
        dice = DiceLoss()

        self.z_normalize = True
        self.loss_function = weighted_pixelwise_cross_entropy
        self.visualize_every_n_epochs = 10
    
    def forward(self, x):        
        return self.model(x)

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
            val=False,
            real_transform=transform_r,
            imag_transform=transform_i)

        self.val_dataset = OPH_Dataset(
            validation_index=self.validation_index,
            val=True,
            real_transform=transform_r,
            imag_transform=transform_i)

        visualization_dataset = OPH_VisualizationDataset(
            real_transform=transform_r,
            imag_transform=transform_i)

        self.visualization_dataloader = DataLoader(visualization_dataset, batch_size=106, shuffle=False)

    def train_dataloader(self):
        return DataLoader(self.train_dataset, batch_size=64, shuffle=True)
        
    def val_dataloader(self):        
        return DataLoader(self.val_dataset, batch_size=64, shuffle=False)
        
    def training_step(self, batch, batch_idx):
        data, label = batch
        data = transform_complex_to_real(data)

        data, label = data.double(), label.double()
        prediction = self(data)
        loss = self.loss_function(prediction, label)

        self.log('train/loss', loss)
        return loss

    def validation_step(self, batch, batch_idx):
        data, label = batch
        data = transform_complex_to_real(data)

        prediction = self(data)
        loss = self.loss_function(prediction, label)        
    
        self.log('val/loss', loss)
        return loss

    def predict_oph(self):
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

    def visualize_progress(self):
        image = self.predict_oph()
        tensorboard = self.logger.experiment
        tensorboard.add_image("Prediction", image, global_step = self.epoch, dataformats='HWC')

    def on_validation_epoch_end(self):
        self.epoch += 1
        if self.epoch % self.visualize_every_n_epochs == 1:
            self.visualize_progress()
        return super().on_validation_epoch_end()

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer

if __name__ == '__main__':
    model = Polsarnet()
    trainer = pl.Trainer(gpus=1)
    trainer.fit(model)
