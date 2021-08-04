import torch
from torch import nn
from torch import Tensor

class ComplexConv1d(nn.Conv1d):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)

    def forward(self, x) -> torch.Tensor:
        _, filter, *_ = x.shape
        x_real, x_imag =  x[:,:filter//2], x[:,filter//2:]
        y_real = super().forward(x_real)
        y_imag = super().forward(x_imag) 
        return torch.hstack((y_real, y_imag))

class ComplexConv2d(nn.Conv2d):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)

    def forward(self, x) -> torch.Tensor:
        _, filter, *_ = x.shape
        x_real, x_imag =  x[:,:filter//2], x[:,filter//2:]
        y_real = super().forward(x_real)
        y_imag = super().forward(x_imag) 
        return torch.hstack((y_real, y_imag))

class ComplexBatchNorm1d(nn.BatchNorm1d):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x) -> torch.Tensor:
        _, filter, *_ = x.shape
        x_real, x_imag =  x[:,:filter//2], x[:,filter//2:]
        y_real = super().forward(x_real)
        y_imag = super().forward(x_imag) 
        return torch.hstack((y_real, y_imag))

class ComplexBatchNorm2d(nn.BatchNorm2d):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
    
    def forward(self, x) -> torch.Tensor:
        _, filter, *_ = x.shape
        x_real, x_imag =  x[:,:filter//2], x[:,filter//2:]
        y_real = super().forward(x_real)
        y_imag = super().forward(x_imag) 
        return torch.hstack((y_real, y_imag))

class ComplexRelu(nn.ReLU):

    def __init__(self, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)

    def forward(self, x: Tensor) -> Tensor:
        _, filter, *_ = x.shape
        x_real, x_imag =  x[:,:filter//2], x[:,filter//2:]
        y_real = super().forward(x_real)
        y_imag = super().forward(x_imag) 
        return torch.hstack((y_real, y_imag))

class ComplexSigmoid(nn.Sigmoid):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        _, filter, *_ = x.shape
        x_real, x_imag =  x[:,:filter//2], x[:,filter//2:]
        y_real = super().forward(x_real)
        y_imag = super().forward(x_imag) 
        return torch.hstack((y_real, y_imag))

class ComplexSoftmax(nn.Softmax):
    
    def __init__(self, *args, **kwargs) -> None:
        super().__init__( *args, **kwargs)
        
    def forward(self, x: Tensor) -> Tensor:
        _, filter, *_ = x.shape
        x_real, x_imag =  x[:,:filter//2], x[:,filter//2:]
        squared_magnitude = torch.sqrt(x_real * x_real + x_imag * x_imag)
        return super().forward(squared_magnitude)

class PixelwiseCrossEntropy(nn.Module):
    def __init__(self, weights=None, one_hot_encoded=True):
        self.one_hot_encoded = one_hot_encoded
        if weights is not None:
            self.weights = torch.Tensor(weights).to('cuda')
        else:
            self.weights = None
        super(PixelwiseCrossEntropy, self).__init__()

    def forward(self, prediction, label):
        prediction = prediction + 1e-15
        prediction = torch.log(prediction)

        if self.one_hot_encoded:
            label = torch.argmax(label, dim=1)
        
        label = label.long()

        if self.weights is not None:
            error = torch.nn.functional.nll_loss(
                prediction,
                label,
                weight = self.weights 
            )
        else:
            error = torch.nn.functional.nll_loss(
                prediction,
                label
            )

        return error