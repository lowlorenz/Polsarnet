![Oberpfaffenhofen prediction](https://raw.githubusercontent.com/lowlorenz/Polsarnet/main/ressources/oph_prediction.png)

This repository contains an implementation of [Polsarnet](https://ieeexplore.ieee.org/abstract/document/8936481), which was created in the context of the module 'Hot topics in computer vision' at TU Berlin.

The goal of the project is to evaluate the Paper `PolSARNet: A Deep Fully Convolutional Network for Polarimetric SAR Image Classification`. Therefore the implementation is tested on two POLSAR datasets.

These datasets are:
<li> 
The L-band POLTOM dataset of the region Oberpfaffenhofen, as used in <a href="https://ieeexplore.ieee.org/document/8438032">Exploiting GAN-Based SAR to Optical Image Transcoding for Improved Classification via Deep Learning</a>()
<li> The Sentinel-1 radar images of the city Wroclaw (Poland), as used in <a href="http://www.mdpi.com/2072-4292/10/11/1742">Exploiting SAR Tomography for Supervised Land-Cover Classification</a>


A list of dependencies is found in `requirements.txt`.

The dataloader package contains all the files needed for loading these two datasets.

The models package contains the implementation of the networks (`Polsarnet.py`) and the implementation of the complex building blocks (`complexNetwork.py`).

The code used for the metrics is found in `evaluation.py`. 

The training is done by calling either `oberpfaffenhofen.py` or `poland.py`.

