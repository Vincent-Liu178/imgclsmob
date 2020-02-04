# Computer vision models on TensorFlow 2.x

[![PyPI](https://img.shields.io/pypi/v/tf2cv.svg)](https://pypi.python.org/pypi/tf2cv)
[![Downloads](https://pepy.tech/badge/tf2cv)](https://pepy.tech/project/tf2cv)

This is a collection of image classification models. Many of them are pretrained on
[ImageNet-1K](http://www.image-net.org), [CIFAR-10/100](https://www.cs.toronto.edu/~kriz/cifar.html), and
[SVHN](http://ufldl.stanford.edu/housenumbers) datasets and loaded automatically during use. All pretrained models
require the same ordinary normalization. Scripts for training/evaluating/converting models are in the
[`imgclsmob`](https://github.com/osmr/imgclsmob) repo.

## List of implemented models

- AlexNet (['One weird trick for parallelizing convolutional neural networks'](https://arxiv.org/abs/1404.5997))
- ZFNet (['Visualizing and Understanding Convolutional Networks'](https://arxiv.org/abs/1311.2901))
- VGG/BN-VGG (['Very Deep Convolutional Networks for Large-Scale Image Recognition'](https://arxiv.org/abs/1409.1556))
- BN-Inception (['Batch Normalization: Accelerating Deep Network Training by Reducing Internal Covariate Shift'](https://arxiv.org/abs/1502.03167))
- ResNet (['Deep Residual Learning for Image Recognition'](https://arxiv.org/abs/1512.03385))
- PreResNet (['Identity Mappings in Deep Residual Networks'](https://arxiv.org/abs/1603.05027))
- ResNeXt (['Aggregated Residual Transformations for Deep Neural Networks'](http://arxiv.org/abs/1611.05431))
- SENet/SE-ResNet/SE-PreResNet/SE-ResNeXt (['Squeeze-and-Excitation Networks'](https://arxiv.org/abs/1709.01507))
- IBN-ResNet/IBN-ResNeXt/IBN-DenseNet (['Two at Once: Enhancing Learning and Generalization Capacities via IBN-Net'](https://arxiv.org/abs/1807.09441))
- AirNet/AirNeXt (['Attention Inspiring Receptive-Fields Network for Learning Invariant Representations'](https://ieeexplore.ieee.org/document/8510896))
- BAM-ResNet (['BAM: Bottleneck Attention Module'](https://arxiv.org/abs/1807.06514))
- CBAM-ResNet (['CBAM: Convolutional Block Attention Module'](https://arxiv.org/abs/1807.06521))
- PyramidNet (['Deep Pyramidal Residual Networks'](https://arxiv.org/abs/1610.02915))
- DiracNetV2 (['DiracNets: Training Very Deep Neural Networks Without Skip-Connections'](https://arxiv.org/abs/1706.00388))
- DenseNet (['Densely Connected Convolutional Networks'](https://arxiv.org/abs/1608.06993))
- PeleeNet (['Pelee: A Real-Time Object Detection System on Mobile Devices'](https://arxiv.org/abs/1804.06882))
- WRN (['Wide Residual Networks'](https://arxiv.org/abs/1605.07146))
- DRN-C/DRN-D (['Dilated Residual Networks'](https://arxiv.org/abs/1705.09914))
- DPN (['Dual Path Networks'](https://arxiv.org/abs/1707.01629))
- DarkNet Ref/Tiny/19 (['Darknet: Open source neural networks in c'](https://github.com/pjreddie/darknet))
- DarkNet-53 (['YOLOv3: An Incremental Improvement'](https://arxiv.org/abs/1804.02767))
- BagNet (['Approximating CNNs with Bag-of-local-Features models works surprisingly well on ImageNet'](https://openreview.net/pdf?id=SkfMWhAqYQ))
- HRNet (['Deep High-Resolution Representation Learning for Visual Recognition'](https://arxiv.org/abs/1908.07919))
- VoVNet (['An Energy and GPU-Computation Efficient Backbone Network for Real-Time Object Detection'](https://arxiv.org/abs/1904.09730))
- SelecSLS (['XNect: Real-time Multi-person 3D Human Pose Estimation with a Single RGB Camera'](https://arxiv.org/abs/1907.00837))
- HarDNet (['HarDNet: A Low Memory Traffic Network'](https://arxiv.org/abs/1909.00948))
- SqueezeNet/SqueezeResNet (['SqueezeNet: AlexNet-level accuracy with 50x fewer parameters and <0.5MB model size'](https://arxiv.org/abs/1602.07360))
- SqueezeNext (['SqueezeNext: Hardware-Aware Neural Network Design'](https://arxiv.org/abs/1803.10615))
- ShuffleNet (['ShuffleNet: An Extremely Efficient Convolutional Neural Network for Mobile Devices'](https://arxiv.org/abs/1707.01083))
- ShuffleNetV2 (['ShuffleNet V2: Practical Guidelines for Efficient CNN Architecture Design'](https://arxiv.org/abs/1807.11164))
- MENet (['Merging and Evolution: Improving Convolutional Neural Networks for Mobile Applications'](https://arxiv.org/abs/1803.09127))
- MobileNet (['MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications'](https://arxiv.org/abs/1704.04861))
- FD-MobileNet (['FD-MobileNet: Improved MobileNet with A Fast Downsampling Strategy'](https://arxiv.org/abs/1802.03750))
- MobileNetV2 (['MobileNetV2: Inverted Residuals and Linear Bottlenecks'](https://arxiv.org/abs/1801.04381))
- MobileNetV3 (['Searching for MobileNetV3'](https://arxiv.org/abs/1905.02244))
- IGCV3 (['IGCV3: Interleaved Low-Rank Group Convolutions for Efficient Deep Neural Networks'](https://arxiv.org/abs/1806.00178))
- GhostNet (['GhostNet: More Features from Cheap Operations'](https://arxiv.org/abs/1911.11907))
- MnasNet (['MnasNet: Platform-Aware Neural Architecture Search for Mobile'](https://arxiv.org/abs/1807.11626))
- ProxylessNAS (['ProxylessNAS: Direct Neural Architecture Search on Target Task and Hardware'](https://arxiv.org/abs/1812.00332))
- FBNet (['FBNet: Hardware-Aware Efficient ConvNet Design via Differentiable Neural Architecture Search'](https://arxiv.org/abs/1812.03443))
- Xception (['Xception: Deep Learning with Depthwise Separable Convolutions'](https://arxiv.org/abs/1610.02357))
- InceptionV3 (['Rethinking the Inception Architecture for Computer Vision'](https://arxiv.org/abs/1512.00567))
- InceptionV4/InceptionResNetV2 (['Inception-v4, Inception-ResNet and the Impact of Residual Connections on Learning'](https://arxiv.org/abs/1602.07261))
- PolyNet (['PolyNet: A Pursuit of Structural Diversity in Very Deep Networks'](https://arxiv.org/abs/1611.05725))
- NASNet (['Learning Transferable Architectures for Scalable Image Recognition'](https://arxiv.org/abs/1707.07012))
- PNASNet (['Progressive Neural Architecture Search'](https://arxiv.org/abs/1712.00559))
- SPNASNet (['Single-Path NAS: Designing Hardware-Efficient ConvNets in less than 4 Hours'](https://arxiv.org/abs/1904.02877))
- EfficientNet (['EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks'](https://arxiv.org/abs/1905.11946))
- MixNet (['MixConv: Mixed Depthwise Convolutional Kernels'](https://arxiv.org/abs/1907.09595))

## Installation

To use the models in your project, simply install the `tf2cv` package with `tensorflow`:
```
pip install tf2cv tensorflow>=2.0.0
```
To enable/disable different hardware supports, check out TensorFlow installation [instructions](https://www.tensorflow.org).

## Usage

Example of using a pretrained ResNet-18 model (with `channels_first` data format):
```
from tf2cv.model_provider import get_model as tf2cv_get_model
import tensorflow as tf

net = tf2cv_get_model("resnet18", pretrained=True, data_format="channels_last")
x = tf.random.normal((1, 224, 224, 3))
y_net = net(x)
```

## Pretrained models (ImageNet-1K)

Some remarks:
- Top1/Top5 are the standard 1-crop Top-1/Top-5 errors (in percents) on the validation subset of the ImageNet-1K dataset.
- FLOPs/2 is the number of FLOPs divided by two to be similar to the number of MACs.
- Remark `Converted from GL model` means that the model was trained on `MXNet/Gluon` and then converted to TensorFlow.
- Models with *-suffix use non-standard preprocessing (see the training log).

| Model | Top1 | Top5 | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | ---: | --- |
| AlexNet | 40.50 | 17.89 | 62,378,344 | 1,132.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/alexnet-1789-ecc4bb4e.tf2.h5.log)) |
| AlexNet-b | 41.03 | 18.59 | 61,100,840 | 714.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/alexnetb-1859-9e390537.tf2.h5.log)) |
| ZFNet | 395.0 | 17.17 | 62,357,608 | 1,170.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/zfnet-1717-9500db30.tf2.h5.log)) |
| ZFNet-b | 36.28 | 14.80 | 107,627,624 | 2,479.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/zfnetb-1480-47533f6a.tf2.h5.log)) |
| VGG-11 | 29.59 | 10.17 | 132,863,336 | 7,615.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/vgg11-1017-c20556f4.tf2.h5.log)) |
| VGG-13 | 28.41 | 9.51 | 133,047,848 | 11,317.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/vgg13-0951-9fa609fc.tf2.h5.log)) |
| VGG-16 | 26.59 | 8.34 | 138,357,544 | 15,480.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/vgg16-0834-ce78831f.tf2.h5.log)) |
| VGG-19 | 25.57 | 7.68 | 143,667,240 | 19,642.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/vgg19-0768-ec5ac0ba.tf2.h5.log)) |
| BN-VGG-11 | 28.57 | 9.36 | 132,866,088 | 7,630.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg11-0936-ef31b866.tf2.h5.log)) |
| BN-VGG-13 | 27.67 | 8.87 | 133,050,792 | 11,341.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg13-0887-2cccc725.tf2.h5.log)) |
| BN-VGG-16 | 25.46 | 7.59 | 138,361,768 | 15,506.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg16-0759-1ca9dee8.tf2.h5.log)) |
| BN-VGG-19 | 23.89 | 6.88 | 143,672,744 | 19,671.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg19-0688-81d25be8.tf2.h5.log)) |
| BN-VGG-11b | 29.31 | 9.75 | 132,868,840 | 7,630.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg11b-0975-aeaccfdc.tf2.h5.log)) |
| BN-VGG-13b | 29.46 | 10.19 | 133,053,736 | 11,342.14M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg13b-1019-1102ffb7.tf2.h5.log)) |
| BN-VGG-16b | 26.89 | 8.62 | 138,365,992 | 15,507.20M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg16b-0862-137178f7.tf2.h5.log)) |
| BN-VGG-19b | 25.64 | 8.17 | 143,678,248 | 19,672.26M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/bn_vgg19b-0817-cd68a741.tf2.h5.log)) |
| BN-Inception | 26.62 | 8.65 | 11,295,240 | 2,048.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.423/bninception-0865-4cab3cce.tf2.h5.log)) |
| ResNet-10 | 34.68 | 13.90 | 5,418,792 | 894.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet10-1390-9e787f63.tf2.h5.log)) |
| ResNet-12 | 33.43 | 13.01 | 5,492,776 | 1,126.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet12-1301-8bc41d1b.tf2.h5.log)) |
| ResNet-14 | 32.21 | 12.24 | 5,788,200 | 1,357.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet14-1224-7573d988.tf2.h5.log)) |
| ResNet-BC-14b | 30.21 | 11.15 | 10,064,936 | 1,479.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnetbc14b-1115-5f30b798.tf2.h5.log)) |
| ResNet-16 | 30.22 | 10.88 | 6,968,872 | 1,589.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet16-1088-14ce0d64.tf2.h5.log)) |
| ResNet-18 x0.25 | 39.30 | 17.45 | 3,937,400 | 270.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet18_wd4-1745-6e800416.tf2.h5.log)) |
| ResNet-18 x0.5 | 33.40 | 12.83 | 5,804,296 | 608.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet18_wd2-1283-85a7caff.tf2.h5.log)) |
| ResNet-18 x0.75 | 29.98 | 10.67 | 8,476,056 | 1,129.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet18_w3d4-1067-c1735b7d.tf2.h5.log)) |
| ResNet-18 | 28.10 | 9.56 | 11,689,512 | 1,820.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet18-0956-6645845a.tf2.h5.log)) |
| ResNet-26 | 26.15 | 8.37 | 17,960,232 | 2,746.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet26-0837-a8f20f71.tf2.h5.log)) |
| ResNet-BC-26b | 24.80 | 7.57 | 15,995,176 | 2,356.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnetbc26b-0757-d70a2cad.tf2.h5.log)) |
| ResNet-34 | 24.50 | 7.44 | 21,797,672 | 3,672.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet34-0744-7f7d70e7.tf2.h5.log)) |
| ResNet-BC-38b | 23.44 | 6.77 | 21,925,416 | 3,234.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnetbc38b-0677-75e405a7.tf2.h5.log)) |
| ResNet-50 | 22.09 | 6.04 | 25,557,032 | 3,877.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet50-0604-728800bf.tf2.h5.log)) |
| ResNet-50b | 22.09 | 6.14 | 25,557,032 | 4,110.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet50b-0614-b2a49da6.tf2.h5.log)) |
| ResNet-101 | 21.59 | 6.01 | 44,549,160 | 7,597.95M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet101-0601-b6befeb4.tf2.h5.log)) |
| ResNet-101b | 20.25 | 5.11 | 44,549,160 | 7,830.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet101b-0511-e3076227.tf2.h5.log)) |
| ResNet-152 | 20.72 | 5.34 | 60,192,808 | 11,321.85M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet152-0534-2d8e394a.tf2.h5.log)) |
| ResNet-152b | 19.60 | 4.80 | 60,192,808 | 11,554.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnet152b-0480-b77f1e2c.tf2.h5.log)) |
| PreResNet-10 | 34.71 | 14.02 | 5,417,128 | 894.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet10-1402-541bf0e1.tf2.h5.log)) |
| PreResNet-12 | 33.63 | 13.20 | 5,491,112 | 1,126.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet12-1320-349c0df4.tf2.h5.log)) |
| PreResNet-14 | 32.29 | 12.24 | 5,786,536 | 1,358.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet14-1224-194b8762.tf2.h5.log)) |
| PreResNet-BC-14b | 30.73 | 11.52 | 10,057,384 | 1,476.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnetbc14b-1152-bc4e06ff.tf2.h5.log)) |
| PreResNet-16 | 30.17 | 10.80 | 6,967,208 | 1,589.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet16-1080-e00c40ee.tf2.h5.log)) |
| PreResNet-18 x0.25 | 39.61 | 17.80 | 3,935,960 | 270.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet18_wd4-1780-6ac7bc59.tf2.h5.log)) |
| PreResNet-18 x0.5 | 33.70 | 13.14 | 5,802,440 | 608.73M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet18_wd2-1314-0c0528c8.tf2.h5.log)) |
| PreResNet-18 x0.75 | 29.95 | 10.70 | 8,473,784 | 1,129.51M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet18_w3d4-1070-056b46c6.tf2.h5.log)) |
| PreResNet-18 | 28.20 | 9.55 | 11,687,848 | 1,820.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet18-0955-621ead92.tf2.h5.log)) |
| PreResNet-26 | 25.98 | 8.37 | 17,958,568 | 2,746.94M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet26-0837-1a92a732.tf2.h5.log)) |
| PreResNet-BC-26b | 25.22 | 7.88 | 15,987,624 | 2,354.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnetbc26b-0788-1f737cd6.tf2.h5.log)) |
| PreResNet-34 | 24.60 | 7.54 | 21,796,008 | 3,672.83M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet34-0754-3cc5ae14.tf2.h5.log)) |
| PreResNet-BC-38b | 22.70 | 6.36 | 21,917,864 | 3,231.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnetbc38b-0636-3396b49b.tf2.h5.log)) |
| PreResNet-50 | 22.22 | 6.25 | 25,549,480 | 3,875.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet50-0625-20860562.tf2.h5.log)) |
| PreResNet-50b | 22.37 | 6.34 | 25,549,480 | 4,107.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet50b-0634-711227b1.tf2.h5.log)) |
| PreResNet-101 | 21.47 | 5.73 | 44,541,608 | 7,595.44M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet101-0573-d45ea488.tf2.h5.log)) |
| PreResNet-101b | 20.86 | 5.39 | 44,541,608 | 7,827.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet101b-0539-54d23aff.tf2.h5.log)) |
| PreResNet-152 | 20.71 | 5.32 | 60,185,256 | 11,319.34M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet152-0532-0ad4b58f.tf2.h5.log)) |
| PreResNet-152b | 19.86 | 5.00 | 60,185,256 | 11,551.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet152b-0500-119062d9.tf2.h5.log)) |
| PreResNet-200b | 21.07 | 5.63 | 64,666,280 | 15,068.63M | From [tornadomeet/ResNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet200b-0563-2f9c761d.tf2.h5.log)) |
| PreResNet-269b | 20.75 | 5.57 | 102,065,832 | 20,101.11M | From [soeaver/mxnet-model] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/preresnet269b-0557-7003b3c4.tf2.h5.log)) |
| ResNeXt-14 (16x4d) | 31.69 | 12.22 | 7,127,336 | 1,045.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext14_16x4d-1222-bff90c1d.tf2.h5.log)) |
| ResNeXt-14 (32x2d) | 32.14 | 12.47 | 7,029,416 | 1,031.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext14_32x2d-1247-06aa6709.tf2.h5.log)) |
| ResNeXt-14 (32x4d) | 29.94 | 11.15 | 9,411,880 | 1,603.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext14_32x4d-1115-3acdaec1.tf2.h5.log)) |
| ResNeXt-26 (32x2d) | 26.32 | 8.51 | 9,924,136 | 1,461.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext26_32x2d-0851-827791cc.tf2.h5.log)) |
| ResNeXt-26 (32x4d) | 23.94 | 7.18 | 15,389,480 | 2,488.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext26_32x4d-0718-4f05525e.tf2.h5.log)) |
| ResNeXt-50 (32x4d) | 20.62 | 5.47 | 25,028,904 | 4,255.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext50_32x4d-0547-45234d14.tf2.h5.log)) |
| ResNeXt-101 (32x4d) | 19.65 | 4.94 | 44,177,704 | 8,003.45M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext101_32x4d-0494-3990ddd1.tf2.h5.log)) |
| ResNeXt-101 (64x4d) | 19.31 | 4.84 | 83,455,272 | 15,500.27M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/resnext101_64x4d-0484-f8cf1580.tf2.h5.log)) |
| SE-ResNet-10 | 33.54 | 13.32 | 5,463,332 | 894.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet10-1332-33a592e1.tf2.h5.log)) |
| SE-ResNet-18 | 27.97 | 9.21 | 11,778,592 | 1,820.88M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet18-0921-46c847ab.tf2.h5.log)) |
| SE-ResNet-26 | 25.42 | 8.07 | 18,093,852 | 2,747.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet26-0807-5178b3b1.tf2.h5.log)) |
| SE-ResNet-BC-26b | 23.39 | 6.84 | 17,395,976 | 2,359.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnetbc26b-0684-1460a381.tf2.h5.log)) |
| SE-ResNet-BC-38b | 21.43 | 5.75 | 24,026,616 | 3,238.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnetbc38b-0575-18fcfcc1.tf2.h5.log)) |
| SE-ResNet-50 | 21.09 | 5.60 | 28,088,024 | 3,883.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.441/seresnet50-0560-f1b84c8d.tf2.h5.log)) |
| SE-ResNet-50b | 20.58 | 5.33 | 28,088,024 | 4,115.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet50b-0533-256002c3.tf2.h5.log)) |
| SE-ResNet-101 | 21.94 | 5.89 | 49,326,872 | 7,602.76M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet101-0589-2a22ba87.tf2.h5.log)) |
| SE-ResNet-152 | 21.47 | 5.76 | 66,821,848 | 11,328.52M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnet152-0576-8023259a.tf2.h5.log)) |
| SE-PreResNet-10 | 33.62 | 13.09 | 5,461,668 | 894.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sepreresnet10-1309-af20d06c.tf2.h5.log)) |
| SE-PreResNet-18 | 27.70 | 9.40 | 11,776,928 | 1,821.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sepreresnet18-0940-fe403280.tf2.h5.log)) |
| SE-PreResNet-BC-26b | 22.95 | 6.40 | 17,388,424 | 2,357.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sepreresnetbc26b-0640-a72bf876.tf2.h5.log)) |
| SE-PreResNet-BC-38b | 21.44 | 5.67 | 24,019,064 | 3,236.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sepreresnetbc38b-0567-17d10c63.tf2.h5.log)) |
| SE-ResNeXt-50 (32x4d) | 19.98 | 5.09 | 27,559,896 | 4,261.16M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnext50_32x4d-0509-4244900a.tf2.h5.log)) |
| SE-ResNeXt-101 (32x4d) | 19.01 | 4.59 | 48,955,416 | 8,012.73M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnext101_32x4d-0459-13a9b2fd.tf2.h5.log)) |
| SE-ResNeXt-101 (64x4d) | 18.96 | 4.65 | 88,232,984 | 15,509.54M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/seresnext101_64x4d-0465-ec0a3b13.tf2.h5.log)) |
| SENet-16 | 25.37 | 8.05 | 31,366,168 | 5,081.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/senet16-0805-f5f57656.tf2.h5.log)) |
| SENet-28 | 21.68 | 5.90 | 36,453,768 | 5,732.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/senet28-0590-667d5687.tf2.h5.log)) |
| SENet-154 | 18.78 | 4.66 | 115,088,984 | 20,745.78M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/senet154-0466-f1b79a9b.tf2.h5.log)) |
| IBN-ResNet-50 | 23.53 | 6.68 | 25,557,032 | 4,110.48M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/ibn_resnet50-0668-4c72a071.tf2.h5.log)) |
| IBN-ResNet-101 | 21.86 | 5.84 | 44,549,160 | 7,830.48M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/ibn_resnet101-0584-2c2c4993.tf2.h5.log)) |
| IBN(b)-ResNet-50 | 23.88 | 6.95 | 25,558,568 | 4,112.89M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/ibnb_resnet50-0695-7178cc50.tf2.h5.log)) |
| IBN-ResNeXt-101 (32x4d) | 21.41 | 5.64 | 44,177,704 | 8,003.45M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/ibn_resnext101_32x4d-0564-c149beb5.tf2.h5.log)) |
| IBN-DenseNet-121 | 24.96 | 7.49 | 7,978,856 | 2,872.13M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/ibn_densenet121-0749-009d1919.tf2.h5.log)) |
| IBN-DenseNet-169 | 23.75 | 6.84 | 14,149,480 | 3,403.89M | From [XingangPan/IBN-Net] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/ibn_densenet169-0684-7152d6cc.tf2.h5.log)) |
| AirNet50-1x64d (r=2) | 22.54 | 6.23 | 27,425,864 | 4,772.11M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.423/airnet50_1x64d_r2-0623-6940f0e5.tf2.h5.log)) |
| AirNet50-1x64d (r=16) | 22.89 | 6.50 | 25,714,952 | 4,399.97M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.423/airnet50_1x64d_r16-0650-b7bb8662.tf2.h5.log)) |
| AirNeXt50-32x4d (r=2) | 21.47 | 5.72 | 27,604,296 | 5,339.58M | From [soeaver/AirNet-PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.423/airnext50_32x4d_r2-0572-fa8e40ab.tf2.h5.log)) |
| BAM-ResNet-50 | 23.67 | 6.97 | 25,915,099 | 4,196.09M | From [Jongchan/attention-module] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/bam_resnet50-0697-3a4101c8.tf2.h5.log)) |
| CBAM-ResNet-50 | 22.96 | 6.39 | 28,089,624 | 4,116.97M | From [Jongchan/attention-module] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/cbam_resnet50-0639-1d0bdb0e.tf2.h5.log)) |
| PyramidNet-101 (a=360) | 22.68 | 6.51 | 42,455,070 | 8,743.54M | From [dyhan0920/Pyramid...PyTorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.423/pyramidnet101_a360-0651-9db84918.tf2.h5.log)) |
| DiracNetV2-18 | 30.59 | 11.13 | 11,511,784 | 1,796.62M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/diracnet18v2-1113-4d687b74.tf2.h5.log)) |
| DiracNetV2-34 | 27.92 | 9.50 | 21,616,232 | 3,646.93M | From [szagoruyko/diracnets] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/diracnet34v2-0950-161d97fd.tf2.h5.log)) |
| DenseNet-121 | 23.23 | 6.84 | 7,978,856 | 2,872.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/densenet121-0684-e9196a9c.tf2.h5.log)) |
| DenseNet-161 | 21.84 | 5.91 | 28,681,000 | 7,793.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.432/densenet161-0591-78224027.tf2.h5.log)) |
| DenseNet-169 | 22.13 | 6.06 | 14,149,480 | 3,403.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/densenet169-0606-f708dc33.tf2.h5.log)) |
| DenseNet-201 | 21.57 | 5.91 | 20,013,928 | 4,347.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.426/densenet201-0591-450c6568.tf2.h5.log)) |
| PeleeNet | 31.65 | 11.29 | 2,802,248 | 514.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/peleenet-1129-e1c3cdea.tf2.h5.log)) |
| WRN-50-2 | 22.10 | 6.14 | 68,849,128 | 11,405.42M | From [szagoruyko/functional-zoo] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.423/wrn50_2-0614-bea17aa9.tf2.h5.log)) |
| DRN-C-26 | 25.70 | 7.88 | 21,126,584 | 16,993.90M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.425/drnc26-0788-571eb2dc.tf2.h5.log)) |
| DRN-C-42 | 23.74 | 6.93 | 31,234,744 | 25,093.75M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.425/drnc42-0693-52dd6028.tf2.h5.log)) |
| DRN-C-58 | 22.36 | 6.26 | 40,542,008 | 32,489.94M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.425/drnc58-0626-e5c7be89.tf2.h5.log)) |
| DRN-D-22 | 26.67 | 8.48 | 16,393,752 | 13,051.33M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.425/drnd22-0848-42f7a37b.tf2.h5.log)) |
| DRN-D-38 | 24.52 | 7.37 | 26,501,912 | 21,151.19M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.425/drnd38-0737-a1108275.tf2.h5.log)) |
| DRN-D-54 | 22.07 | 6.26 | 35,809,176 | 28,547.38M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.425/drnd54-0626-cb792485.tf2.h5.log)) |
| DRN-D-105 | 21.31 | 5.83 | 54,801,304 | 43,442.43M | From [fyu/drn] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.425/drnd105-0583-80eb9ec2.tf2.h5.log)) |
| DPN-68 | 22.92 | 6.58 | 12,611,602 | 2,351.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dpn68-0658-5b70b7b8.tf2.h5.log)) |
| DPN-98 | 20.24 | 5.28 | 61,570,728 | 11,716.51M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dpn98-0528-6883ec37.tf2.h5.log)) |
| DPN-131 | 20.05 | 5.24 | 79,254,504 | 16,076.15M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dpn131-0524-971af47c.tf2.h5.log)) |
| DarkNet Tiny | 40.34 | 17.45 | 1,042,104 | 500.85M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/darknet_tiny-1745-d30be41a.tf2.h5.log)) |
| DarkNet Ref | 38.10 | 16.71 | 7,319,416 | 367.59M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/darknet_ref-1671-b4991f6b.tf2.h5.log)) |
| DarkNet-53 | 21.41 | 5.58 | 41,609,928 | 7,133.86M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/darknet53-0558-4a63ab30.tf2.h5.log)) |
| BagNet-9 | 59.59 | 35.53 | 15,688,744 | 16,049.19M | From [wielandbrendel/bag...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/bagnet9-3553-43eb57dc.tf2.h5.log)) |
| BagNet-17 | 44.75 | 21.54 | 16,213,032 | 15,768.77M | From [wielandbrendel/bag...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/bagnet17-2154-8a31e347.tf2.h5.log)) |
| BagNet-33 | 36.42 | 14.97 | 18,310,184 | 16,371.52M | From [wielandbrendel/bag...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/bagnet33-1497-ef600c89.tf2.h5.log)) |
| DLA-34 | 26.15 | 8.23 | 15,742,104 | 3,071.37M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla34-0823-9232e3e7.tf2.h5.log)) |
| DLA-46-C | 33.83 | 12.87 | 1,301,400 | 585.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla46c-1287-dfcae3b5.tf2.h5.log)) |
| DLA-X-46-C | 32.90 | 12.29 | 1,068,440 | 546.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla46xc-1229-a858beca.tf2.h5.log)) |
| DLA-60 | 23.83 | 7.11 | 22,036,632 | 4,255.49M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla60-0711-7375fcfd.tf2.h5.log)) |
| DLA-X-60 | 22.46 | 6.21 | 17,352,344 | 3,543.68M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla60x-0621-3c5941db.tf2.h5.log)) |
| DLA-X-60-C | 30.66 | 10.75 | 1,319,832 | 596.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla60xc-1075-a7850f03.tf2.h5.log)) |
| DLA-102 | 22.84 | 6.43 | 33,268,888 | 7,190.95M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla102-0643-2be886b2.tf2.h5.log)) |
| DLA-X-102 | 21.95 | 6.02 | 26,309,272 | 5,884.94M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla102x-0602-46640eec.tf2.h5.log)) |
| DLA-X2-102 | 21.11 | 5.53 | 41,282,200 | 9,340.61M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla102x2-0553-06c93031.tf2.h5.log)) |
| DLA-169 | 21.97 | 5.90 | 53,389,720 | 11,593.20M | From [ucbdrive/dla] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/dla169-0590-e010166d.tf2.h5.log)) |
| HRNet-W18 Small V1 | 28.43 | 9.74 | 13,187,464 | 1,614.87M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnet_w18_small_v1-0974-8db99936.tf2.h5.log)) |
| HRNet-W18 Small V2 | 25.72 | 8.05 | 15,597,464 | 2,618.54M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnet_w18_small_v2-0805-fcb8e218.tf2.h5.log)) |
| HRNetV2-W18 | 24.02 | 6.86 | 21,299,004 | 4,322.66M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnetv2_w18-0686-71c614d7.tf2.h5.log)) |
| HRNetV2-W30 | 22.31 | 6.06 | 37,712,220 | 8,156.14M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnetv2_w30-0606-4883e345.tf2.h5.log)) |
| HRNetV2-W32 | 22.32 | 6.07 | 41,232,680 | 8,973.31M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnetv2_w32-0607-ef949840.tf2.h5.log)) |
| HRNetV2-W40 | 21.71 | 5.73 | 57,557,160 | 12,751.34M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnetv2_w40-0573-29cece1c.tf2.h5.log)) |
| HRNetV2-W44 | 21.74 | 5.95 | 67,064,984 | 14,945.95M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnetv2_w44-0595-a4e4781c.tf2.h5.log)) |
| HRNetV2-W48 | 21.42 | 5.81 | 77,469,864 | 17,344.29M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnetv2_w48-0581-3af4ed57.tf2.h5.log)) |
| HRNetV2-W64 | 21.10 | 5.53 | 128,059,944 | 28,974.95M | From [HRNet/HRNet...ation] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/hrnetv2_w64-0553-aede8def.tf2.h5.log)) |
| VoVNet-39 | 26.29 | 8.25 | 22,600,296 | 7,086.16M | From [stigma0617/VoVNet.pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.431/vovnet39-0825-49cbcdc6.tf2.h5.log)) |
| VoVNet-57 | 25.65 | 8.12 | 36,640,296 | 8,943.09M | From [stigma0617/VoVNet.pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.431/vovnet57-0812-0977958a.tf2.h5.log)) |
| SelecSLS-42b | 23.28 | 6.76 | 32,458,248 | 2,980.62M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.430/selecsls42b-0676-0d785bec.tf2.h5.log)) |
| SelecSLS-60 | 22.45 | 6.30 | 30,670,768 | 3,591.78M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.430/selecsls60-0630-a799a0e5.tf2.h5.log)) |
| SelecSLS-60b | 21.89 | 6.04 | 32,774,064 | 3,629.14M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.430/selecsls60b-0604-bc9c4319.tf2.h5.log)) |
| HarDNet-39DS | 28.69 | 10.03 | 3,488,228 | 437.52M | From [PingoLH/Pytorch-HarDNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.435/hardnet39ds-1003-4971cd5a.tf2.h5.log)) |
| HarDNet-68DS | 26.36 | 8.45 | 4,180,602 | 788.86M | From [PingoLH/Pytorch-HarDNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.435/hardnet68ds-0845-dd35f3f9.tf2.h5.log)) |
| HarDNet-68 | 24.55 | 7.40 | 17,565,348 | 4,256.32M | From [PingoLH/Pytorch-HarDNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.435/hardnet68-0740-9ea05e39.tf2.h5.log)) |
| HarDNet-85 | 22.62 | 6.44 | 36,670,212 | 9,088.58M | From [PingoLH/Pytorch-HarDNet] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.435/hardnet85-0644-7892e221.tf2.h5.log)) |
| SqueezeNet v1.0 | 39.23 | 17.60 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/squeezenet_v1_0-1760-d13ba732.tf2.h5.log)) |
| SqueezeNet v1.1 | 39.12 | 17.42 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/squeezenet_v1_1-1742-95b61448.tf2.h5.log)) |
| SqueezeResNet v1.0 | 39.38 | 17.83 | 1,248,424 | 823.67M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/squeezeresnet_v1_0-1783-db620d99.tf2.h5.log)) |
| SqueezeResNet v1.1 | 39.85 | 17.89 | 1,235,496 | 352.02M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/squeezeresnet_v1_1-1789-13d6bc6b.tf2.h5.log)) |
| 1.0-SqNxt-23 | 42.31 | 18.61 | 724,056 | 287.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sqnxt23_w1-1861-379975eb.tf2.h5.log)) |
| 1.0-SqNxt-23v5 | 40.44 | 17.62 | 921,816 | 285.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sqnxt23v5_w1-1762-153b4ce7.tf2.h5.log)) |
| 1.5-SqNxt-23 | 34.62 | 13.34 | 1,511,824 | 552.39M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sqnxt23_w3d2-1334-a2ba956c.tf2.h5.log)) |
| 1.5-SqNxt-23v5 | 33.55 | 12.84 | 1,953,616 | 550.97M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sqnxt23v5_w3d2-1284-72efaa71.tf2.h5.log)) |
| 2.0-SqNxt-23 | 30.12 | 10.69 | 2,583,752 | 898.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sqnxt23_w2-1069-f43dee19.tf2.h5.log)) |
| 2.0-SqNxt-23v5 | 29.40 | 10.26 | 3,366,344 | 897.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/sqnxt23v5_w2-1026-da80c640.tf2.h5.log)) |
| ShuffleNet x0.25 (g=1) | 62.05 | 36.81 | 209,746 | 12.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g1_wd4-3681-04a9e2d4.tf2.h5.log)) |
| ShuffleNet x0.25 (g=3) | 61.31 | 36.18 | 305,902 | 13.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g3_wd4-3618-c9aad0f0.tf2.h5.log)) |
| ShuffleNet x0.5 (g=1) | 46.25 | 22.36 | 534,484 | 41.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g1_wd2-2236-082db702.tf2.h5.log)) |
| ShuffleNet x0.5 (g=3) | 43.84 | 20.59 | 718,324 | 41.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g3_wd2-2059-e3aefeeb.tf2.h5.log)) |
| ShuffleNet x0.75 (g=1) | 39.24 | 16.79 | 975,214 | 86.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g1_w3d4-1679-a1cc5da3.tf2.h5.log)) |
| ShuffleNet x0.75 (g=3) | 37.80 | 16.11 | 1,238,266 | 85.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g3_w3d4-1611-89546a05.tf2.h5.log)) |
| ShuffleNet x1.0 (g=1) | 34.48 | 13.48 | 1,531,936 | 148.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g1_w1-1348-52ddb20f.tf2.h5.log)) |
| ShuffleNet x1.0 (g=2) | 33.95 | 13.33 | 1,733,848 | 147.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g2_w1-1333-2a8ba692.tf2.h5.log)) |
| ShuffleNet x1.0 (g=3) | 33.93 | 13.32 | 1,865,728 | 145.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g3_w1-1326-daaec8b8.tf2.h5.log)) |
| ShuffleNet x1.0 (g=4) | 33.88 | 13.13 | 1,968,344 | 143.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g4_w1-1313-35dbd6b9.tf2.h5.log)) |
| ShuffleNet x1.0 (g=8) | 33.71 | 13.22 | 2,434,768 | 150.76M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenet_g8_w1-1322-449fb276.tf2.h5.log)) |
| ShuffleNetV2 x0.5 | 40.75 | 18.43 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2_wd2-1843-d492d721.tf2.h5.log)) |
| ShuffleNetV2 x1.0 | 31.00 | 11.35 | 2,278,604 | 149.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2_w1-1135-dae13ee9.tf2.h5.log)) |
| ShuffleNetV2 x1.5 | 27.41 | 9.23 | 4,406,098 | 320.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2_w3d2-0923-ea615baa.tf2.h5.log)) |
| ShuffleNetV2 x2.0 | 25.83 | 8.21 | 7,601,686 | 595.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2_w2-0821-6ccac868.tf2.h5.log)) |
| ShuffleNetV2b x0.5 | 39.80 | 17.84 | 1,366,792 | 43.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2b_wd2-1784-d5644a6a.tf2.h5.log)) |
| ShuffleNetV2b x1.0 | 30.36 | 11.04 | 2,279,760 | 150.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2b_w1-1104-b7db0ca0.tf2.h5.log)) |
| ShuffleNetV2b x1.5 | 26.90 | 8.77 | 4,410,194 | 323.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2b_w3d2-0877-9efb13f7.tf2.h5.log)) |
| ShuffleNetV2b x2.0 | 25.24 | 8.08 | 7,611,290 | 603.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/shufflenetv2b_w2-0808-ba5c7ddc.tf2.h5.log)) |
| 108-MENet-8x1 (g=3) | 43.64 | 20.39 | 654,516 | 42.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet108_8x1_g3-2039-1a8cfc92.tf2.h5.log)) |
| 128-MENet-8x1 (g=4) | 42.04 | 19.18 | 750,796 | 45.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet128_8x1_g4-1918-7fb59f0a.tf2.h5.log)) |
| 160-MENet-8x1 (g=8) | 43.48 | 20.34 | 850,120 | 45.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet160_8x1_g8-2034-3cf9eb2a.tf2.h5.log)) |
| 228-MENet-12x1 (g=3) | 33.80 | 12.91 | 1,806,568 | 152.93M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet228_12x1_g3-1291-21bd19bf.tf2.h5.log)) |
| 256-MENet-12x1 (g=4) | 32.28 | 12.17 | 1,888,240 | 150.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet256_12x1_g4-1217-d9f2e10e.tf2.h5.log)) |
| 348-MENet-12x1 (g=3) | 27.81 | 9.37 | 3,368,128 | 312.00M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet348_12x1_g3-0937-cee7691c.tf2.h5.log)) |
| 352-MENet-12x1 (g=8) | 31.33 | 11.67 | 2,272,872 | 157.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet352_12x1_g8-1167-54a916bc.tf2.h5.log)) |
| 456-MENet-24x1 (g=3) | 25.02 | 7.79 | 5,304,784 | 567.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/menet456_24x1_g3-0779-2a70b14b.tf2.h5.log)) |
| MobileNet x0.25 | 45.84 | 22.13 | 470,072 | 44.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenet_wd4-2213-ad04596a.tf2.h5.log)) |
| MobileNet x0.5 | 33.86 | 13.33 | 1,331,592 | 155.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenet_wd2-1333-01395e1b.tf2.h5.log)) |
| MobileNet x0.75 | 29.88 | 10.51 | 2,585,560 | 333.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenet_w3d4-1051-7832561b.tf2.h5.log)) |
| MobileNet x1.0 | 26.45 | 8.66 | 4,231,976 | 579.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenet_w1-0866-6939232b.tf2.h5.log)) |
| FD-MobileNet x0.25 | 55.42 | 30.62 | 383,160 | 12.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/fdmobilenet_wd4-3062-36aa16df.tf2.h5.log)) |
| FD-MobileNet x0.5 | 42.66 | 19.77 | 993,928 | 41.84M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/fdmobilenet_wd2-1977-34541b84.tf2.h5.log)) |
| FD-MobileNet x0.75 | 37.97 | 15.97 | 1,833,304 | 86.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/fdmobilenet_w3d4-1597-0123c031.tf2.h5.log)) |
| FD-MobileNet x1.0 | 33.90 | 13.12 | 2,901,288 | 147.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/fdmobilenet_w1-1312-fa99fb8d.tf2.h5.log)) |
| MobileNetV2 x0.25 | 48.10 | 24.13 | 1,516,392 | 34.24M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenetv2_wd4-2413-c3705f55.tf2.h5.log)) |
| MobileNetV2 x0.5 | 35.62 | 14.46 | 1,964,736 | 100.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenetv2_wd2-1446-b0c9a98b.tf2.h5.log)) |
| MobileNetV2 x0.75 | 29.75 | 10.44 | 2,627,592 | 198.50M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenetv2_w3d4-1044-e122c73e.tf2.h5.log)) |
| MobileNetV2 x1.0 | 26.80 | 8.63 | 3,504,960 | 329.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mobilenetv2_w1-0863-b32cede3.tf2.h5.log)) |
| MobileNetV3 L/224/1.0 | 24.65 | 7.69 | 5,481,752 | 226.80M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.411/mobilenetv3_large_w1-0769-f66596ae.tf2.h5.log)) |
| IGCV3 x0.25 | 53.38 | 28.28 | 1,534,020 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/igcv3_wd4-2828-309359dc.tf2.h5.log)) |
| IGCV3 x0.5 | 39.36 | 17.01 | 1,985,528 | 111.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/igcv3_wd2-1701-b952333a.tf2.h5.log)) |
| IGCV3 x0.75 | 30.74 | 11.00 | 2,638,084 | 210.95M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/igcv3_w3d4-1100-00294c7b.tf2.h5.log)) |
| IGCV3 x1.0 | 27.70 | 8.99 | 3,491,688 | 340.79M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/igcv3_w1-0899-a0cb775d.tf2.h5.log)) |
| MnasNet-B1 | 25.72 | 8.02 | 4,383,312 | 326.30M |  From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mnasnet_b1-0802-763d6849.tf2.h5.log)) |
| MnasNet-A1 | 25.02 | 7.56 | 3,887,038 | 326.07M |  From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.422/mnasnet_a1-0756-8e0f4948.tf2.h5.log)) |
| ProxylessNAS CPU | 24.77 | 7.51 | 4,361,648 | 459.96M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/proxylessnas_cpu-0751-47e14316.tf2.h5.log)) |
| ProxylessNAS GPU | 24.65 | 7.26 | 7,119,848 | 476.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/proxylessnas_gpu-0726-d536cb3e.tf2.h5.log)) |
| ProxylessNAS Mobile | 25.29 | 7.83 | 4,080,512 | 332.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/proxylessnas_mobile-0783-da8cdb80.tf2.h5.log)) |
| ProxylessNAS Mob-14 | 22.93 | 6.53 | 6,857,568 | 597.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.424/proxylessnas_mobile14-0653-478b58cd.tf2.h5.log)) |
| FBNet-Cb | 25.44 | 7.84 | 5,572,200 | 399.26M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/fbnet_cb-0784-acd12097.tf2.h5.log)) |
| Xception | 21.14 | 5.58 | 22,855,952 | 8,403.63M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.423/xception-0558-b95b5051.tf2.h5.log)) |
| InceptionV3 | 21.11 | 5.63 | 23,834,568 | 5,743.06M | From [dmlc/gluon-cv] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/inceptionv3-0563-b0094c1c.tf2.h5.log)) |
| InceptionV4 | 20.78 | 5.41 | 42,679,816 | 12,304.93M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/inceptionv4-0541-c1fa5642.tf2.h5.log)) |
| InceptionResNetV2 | 20.00 | 4.95 | 55,843,464 | 13,188.64M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/inceptionresnetv2-0495-3e2cc545.tf2.h5.log)) |
| PolyNet | 19.09 | 4.51 | 95,366,600 | 34,821.34M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/polynet-0451-e752c86b.tf2.h5.log)) |
| NASNet-A 4@1056 | 25.83 | 8.33 | 5,289,978 | 584.90M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/nasnet_4a1056-0833-9710e638.tf2.h5.log)) |
| NASNet-A 6@4032 | 18.24 | 4.27 | 88,753,150 | 23,976.44M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/nasnet_6a4032-0427-1f0d2198.tf2.h5.log)) |
| PNASNet-5-Large | 18.02 | 4.27 | 86,057,668 | 25,140.77M | From [Cadene/pretrained...pytorch] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.428/pnasnet5large-0427-90e804af.tf2.h5.log)) |
| SPNASNet | 26.97 | 8.73 | 4,421,616 | 346.73M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/spnasnet-0873-a38a57a3.tf2.h5.log)) |
| EfficientNet-B0 | 24.49 | 7.25 | 5,288,548 | 413.13M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/efficientnet_b0-0725-fc13925b.tf2.h5.log)) |
| EfficientNet-B1 | 22.93 | 6.30 | 7,794,184 | 730.44M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/efficientnet_b1-0630-82e0c512.tf2.h5.log)) |
| EfficientNet-B0b | 23.05 | 6.68 | 5,288,548 | 413.13M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b0b-0668-77127244.tf2.h5.log)) |
| EfficientNet-B1b | 21.17 | 5.77 | 7,794,184 | 730.44M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b1b-0577-b294ee16.tf2.h5.log)) |
| EfficientNet-B2b | 20.22 | 5.30 | 9,109,994 | 1,049.29M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b2b-0530-55bcdc5d.tf2.h5.log)) |
| EfficientNet-B3b | 19.14 | 4.69 | 12,233,232 | 1,923.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b3b-0469-b8210e1a.tf2.h5.log)) |
| EfficientNet-B4b | 17.52 | 3.99 | 19,341,616 | 4,597.56M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b4b-0399-5e35e9c5.tf2.h5.log)) |
| EfficientNet-B5b | 16.43 | 3.43 | 30,389,784 | 10,674.67M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b5b-0343-0ed0c69d.tf2.h5.log)) |
| EfficientNet-B6b | 15.96 | 3.12 | 43,040,704 | 19,761.35M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b6b-0312-faf63104.tf2.h5.log)) |
| EfficientNet-B7b | 15.85 | 3.15 | 66,347,960 | 38,949.07M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.429/efficientnet_b7b-0315-4024912e.tf2.h5.log)) |
| EfficientNet-B0c* | 22.62 | 6.46 | 5,288,548 | 414.31M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b0c-0646-2bd0e2af.tf2.h5.log)) |
| EfficientNet-B1c* | 20.98 | 5.82 | 7,794,184 | 732.54M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b1c-0582-a760b325.tf2.h5.log)) |
| EfficientNet-B2c* | 20.21 | 5.33 | 9,109,994 | 1,051.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b2c-0533-ea6ca9cf.tf2.h5.log)) |
| EfficientNet-B3c* | 18.80 | 4.64 | 12,233,232 | 1,928.55M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b3c-0464-1c8fced8.tf2.h5.log)) |
| EfficientNet-B4c* | 17.29 | 3.90 | 19,341,616 | 4,607.46M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b4c-0390-dc4379ea.tf2.h5.log)) |
| EfficientNet-B5c* | 15.87 | 3.10 | 30,389,784 | 10,695.20M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b5c-0310-80258ef7.tf2.h5.log)) |
| EfficientNet-B6c* | 15.29 | 2.86 | 43,040,704 | 19,796.24M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b6c-0286-285f830a.tf2.h5.log)) |
| EfficientNet-B7c* | 14.96 | 2.76 | 66,347,960 | 39,010.98M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b7c-0276-1ffad4ec.tf2.h5.log)) |
| EfficientNet-B8c* | 14.64 | 2.70 | 87,413,142 | 64,541.66M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.433/efficientnet_b8c-0270-aa691b94.tf2.h5.log)) |
| EfficientNet-Edge-Small-b* | 22.66 | 6.42 | 5,438,392 | 2,378.12M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.434/efficientnet_edge_small_b-0642-1c03bb73.tf2.h5.log)) |
| EfficientNet-Edge-Medium-b* | 21.38 | 5.65 | 6,899,496 | 3,700.12M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.434/efficientnet_edge_medium_b-0565-73153b18.tf2.h5.log)) |
| EfficientNet-Edge-Large-b* | 19.86 | 4.96 | 10,589,712 | 9,747.66M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.434/efficientnet_edge_large_b-0496-d72edce1.tf2.h5.log)) |
| MixNet-S | 24.34 | 7.37 | 4,134,606 | 260.26M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/mixnet_s-0737-d68d63f1.tf2.h5.log)) |
| MixNet-M | 23.29 | 6.79 | 5,014,382 | 366.05M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/mixnet_m-0679-f74eab6c.tf2.h5.log)) |
| MixNet-L | 21.57 | 6.01 | 7,329,252 | 590.45M | From [rwightman/pyt...models] ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.427/mixnet_l-0601-5c2ccc0c.tf2.h5.log)) |

### CIFAR-10

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| ResNet-20 | 5.97 | 272,474 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet20_cifar10-0597-451230e9.tf2.h5.log)) |
| ResNet-56 | 4.52 | 855,770 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet56_cifar10-0452-a39ad94a.tf2.h5.log)) |
| ResNet-110 | 3.69 | 1,730,714 | 255.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet110_cifar10-0369-c625643a.tf2.h5.log)) |
| ResNet-164(BN) | 3.68 | 1,704,154 | 255.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet164bn_cifar10-0368-cf08cca7.tf2.h5.log)) |
| ResNet-272(BN) | 3.33 | 2,816,986 | 420.61M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet272bn_cifar10-0333-c8b0a926.tf2.h5.log)) |
| ResNet-542(BN) | 3.43 | 5,599,066 | 833.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet542bn_cifar10-0343-c31829d4.tf2.h5.log)) |
| ResNet-1001 | 3.28 | 10,328,602 | 1,536.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet1001_cifar10-0328-552ab287.tf2.h5.log)) |
| ResNet-1202 | 3.53 | 19,424,026 | 2,857.17M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet1202_cifar10-0353-3559a943.tf2.h5.log)) |
| PreResNet-20 | 6.51 | 272,282 | 41.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet20_cifar10-0651-d3e7771e.tf2.h5.log)) |
| PreResNet-56 | 4.49 | 855,578 | 127.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet56_cifar10-0449-b4bfdaa8.tf2.h5.log)) |
| PreResNet-110 | 3.86 | 1,730,522 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet110_cifar10-0386-287a4b0c.tf2.h5.log)) |
| PreResNet-164(BN) | 3.64 | 1,703,258 | 255.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet164bn_cifar10-0364-29a459fa.tf2.h5.log)) |
| PreResNet-272(BN) | 3.25 | 2,816,090 | 420.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet272bn_cifar10-0325-5bacdc95.tf2.h5.log)) |
| PreResNet-542(BN) | 3.14 | 5,598,170 | 833.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet542bn_cifar10-0314-d8324d47.tf2.h5.log)) |
| PreResNet-1001 | 2.65 | 10,327,706 | 1,536.18M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet1001_cifar10-0265-978844c1.tf2.h5.log)) |
| PreResNet-1202 | 3.39 | 19,423,834 | 2,857.14M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet1202_cifar10-0339-ab04c456.tf2.h5.log)) |
| ResNeXt-20 (1x64d) | 4.33 | 3,446,602 | 538.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_1x64d_cifar10-0433-e0ab8667.tf2.h5.log)) |
| ResNeXt-20 (2x32d) | 4.53 | 2,672,458 | 425.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_2x32d_cifar10-0453-7aa966dd.tf2.h5.log)) |
| ResNeXt-20 (4x16d) | 4.70 | 2,285,386 | 368.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_4x16d_cifar10-0470-333e834d.tf2.h5.log)) |
| ResNeXt-20 (8x8d) | 4.66 | 2,091,850 | 340.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_8x8d_cifar10-0466-1dbd9f5e.tf2.h5.log)) |
| ResNeXt-20 (16x4d) | 4.04 | 1,995,082 | 326.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_16x4d_cifar10-0404-c6719935.tf2.h5.log)) |
| ResNeXt-20 (32x2d) | 4.61 | 1,946,698 | 319.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_32x2d_cifar10-0461-b05d3491.tf2.h5.log)) |
| ResNeXt-20 (64x1d) | 4.93 | 1,922,506 | 315.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x1d_cifar10-0493-a13300ce.tf2.h5.log)) |
| ResNeXt-20 (2x64d) | 4.03 | 6,198,602 | 987.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_2x64d_cifar10-0403-367377ed.tf2.h5.log)) |
| ResNeXt-20 (4x32d) | 3.73 | 4,650,314 | 761.57M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_4x32d_cifar10-0373-e4aa1b0d.tf2.h5.log)) |
| ResNeXt-20 (8x16d) | 4.04 | 3,876,170 | 648.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_8x16d_cifar10-0404-5329db5f.tf2.h5.log)) |
| ResNeXt-20 (16x8d) | 3.94 | 3,489,098 | 591.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_16x8d_cifar10-0394-cf7c675c.tf2.h5.log)) |
| ResNeXt-20 (32x4d) | 4.20 | 3,295,562 | 563.47M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_32x4d_cifar10-0420-6011e9e9.tf2.h5.log)) |
| ResNeXt-20 (64x2d) | 4.38 | 3,198,794 | 549.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x2d_cifar10-0438-3846d7a7.tf2.h5.log)) |
| ResNeXt-56 (1x64d) | 2.87 | 9,317,194 | 1,399.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_1x64d_cifar10-0287-5da5fe18.tf2.h5.log)) |
| ResNeXt-56 (2x32d) | 3.01 | 6,994,762 | 1,059.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_2x32d_cifar10-0301-54d6f2df.tf2.h5.log)) |
| ResNeXt-56 (4x16d) | 3.11 | 5,833,546 | 889.91M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_4x16d_cifar10-0311-766ab89f.tf2.h5.log)) |
| ResNeXt-56 (8x8d) | 3.07 | 5,252,938 | 805.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_8x8d_cifar10-0307-685eab39.tf2.h5.log)) |
| ResNeXt-56 (16x4d) | 3.12 | 4,962,634 | 762.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_16x4d_cifar10-0312-930e5d5b.tf2.h5.log)) |
| ResNeXt-56 (32x2d) | 3.14 | 4,817,482 | 741.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_32x2d_cifar10-0314-9e387e2e.tf2.h5.log)) |
| ResNeXt-56 (64x1d) | 3.41 | 4,744,906 | 730.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_64x1d_cifar10-0341-bc746947.tf2.h5.log)) |
| ResNeXt-29 (32x4d) | 3.15 | 4,775,754 | 780.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x2d_cifar10-0438-3846d7a7.tf2.h5.log)) |
| ResNeXt-29 (16x64d) | 2.41 | 68,155,210 | 10,709.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext29_16x64d_cifar10-0241-712e4744.tf2.h5.log)) |
| ResNeXt-272 (1x64d) | 2.55 | 44,540,746 | 6,565.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext272_1x64d_cifar10-0255-6efe448a.tf2.h5.log)) |
| ResNeXt-272 (2x32d) | 2.74 | 32,928,586 | 4,867.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext272_2x32d_cifar10-0274-4e35f994.tf2.h5.log)) |
| SE-ResNet-20 | 6.01 | 274,847 | 41.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet20_cifar10-0601-2f392e4a.tf2.h5.log)) |
| SE-ResNet-56 | 4.13 | 862,889 | 127.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet56_cifar10-0413-0224e930.tf2.h5.log)) |
| SE-ResNet-110 | 3.63 | 1,744,952 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet110_cifar10-0363-4c28f93f.tf2.h5.log)) |
| SE-ResNet-164(BN) | 3.39 | 1,906,258 | 256.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet164bn_cifar10-0339-64d05154.tf2.h5.log)) |
| SE-ResNet-272(BN) | 3.39 | 3,153,826 | 422.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet272bn_cifar10-0339-baa561b6.tf2.h5.log)) |
| SE-ResNet-542(BN) | 3.47 | 6,272,746 | 838.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet542bn_cifar10-0347-e95ebdb9.tf2.h5.log)) |
| SE-PreResNet-20 | 6.18 | 274,559 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet20_cifar10-0618-22217b32.tf2.h5.log)) |
| SE-PreResNet-56 | 4.51 | 862,601 | 127.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet56_cifar10-0451-32637db5.tf2.h5.log)) |
| SE-PreResNet-110 | 4.54 | 1,744,664 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet110_cifar10-0454-e317c569.tf2.h5.log)) |
| SE-PreResNet-164(BN) | 3.73 | 1,904,882 | 256.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet164bn_cifar10-0373-253c0430.tf2.h5.log)) |
| SE-PreResNet-272(BN) | 3.39 | 3,152,450 | 422.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet272bn_cifar10-0339-1ca0bed3.tf2.h5.log)) |
| SE-PreResNet-542(BN) | 3.08 | 6,271,370 | 837.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet542bn_cifar10-0309-7764e8bd.tf2.h5.log)) |
| PyramidNet-110 (a=48) | 3.72 | 1,772,706 | 408.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a48_cifar10-0372-3b6ab160.tf2.h5.log)) |
| PyramidNet-110 (a=84) | 2.98 | 3,904,446 | 778.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a84_cifar10-0298-bf303f34.tf2.h5.log)) |
| PyramidNet-110 (a=270) | 2.51 | 28,485,477 | 4,730.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a270_cifar10-0251-983d9983.tf2.h5.log)) |
| PyramidNet-164 (a=270, BN) | 2.42 | 27,216,021 | 4,608.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet164_a270_bn_cifar10-0242-aa879193.tf2.h5.log)) |
| PyramidNet-200 (a=240, BN) | 2.44 | 26,752,702 | 4,563.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet200_a240_bn_cifar10-0244-c269bf7d.tf2.h5.log)) |
| PyramidNet-236 (a=220, BN) | 2.47 | 26,969,046 | 4,631.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet236_a220_bn_cifar10-0247-26aac5d0.tf2.h5.log)) |
| PyramidNet-272 (a=200, BN) | 2.39 | 26,210,842 | 4,541.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet272_a200_bn_cifar10-0239-b57f64f1.tf2.h5.log)) |
| DenseNet-40 (k=12) | 5.61 | 599,050 | 210.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k12_cifar10-0561-e6e20ebf.tf2.h5.log)) |
| DenseNet-BC-40 (k=12) | 6.43 | 176,122 | 74.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k12_bc_cifar10-0643-58950791.tf2.h5.log)) |
| DenseNet-BC-40 (k=24) | 4.52 | 690,346 | 293.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k24_bc_cifar10-0452-61a7fe9c.tf2.h5.log)) |
| DenseNet-BC-40 (k=36) | 4.04 | 1,542,682 | 654.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k36_bc_cifar10-0404-ce27624f.tf2.h5.log)) |
| DenseNet-100 (k=12) | 3.66 | 4,068,490 | 1,353.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k12_cifar10-fc483c0b.tf2.h5.log)) |
| DenseNet-100 (k=24) | 3.13 | 16,114,138 | 5,354.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k24_cifar10-0313-7f9ee9b3.tf2.h5.log)) |
| DenseNet-BC-100 (k=12) | 4.16 | 769,162 | 298.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k12_bc_cifar10-0416-66beb8fc.tf2.h5.log)) |
| DenseNet-BC-190 (k=40) | 2.52 | 25,624,430 | 9,400.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet190_k40_bc_cifar10-0252-9cc5cfcb.tf2.h5.log)) |
| DenseNet-BC-250 (k=24) | 2.67 | 15,324,406 | 5,519.54M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet250_k24_bc_cifar10-0267-3217a1b3.tf2.h5.log)) |

### CIFAR-100

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| ResNet-20 | 29.64 | 278,324 | 41.30M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet20_cifar100-2964-5fa28f78.tf2.h5.log)) |
| ResNet-56 | 24.88 | 861,620 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet56_cifar100-2488-8e413ab9.tf2.h5.log)) |
| ResNet-110 | 22.80 | 1,736,564 | 255.71M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet110_cifar100-2280-c248211b.tf2.h5.log)) |
| ResNet-164(BN) | 20.44 | 1,727,284 | 255.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet164bn_cifar100-2044-1ba34790.tf2.h5.log)) |
| ResNet-272(BN) | 20.07 | 2,840,116 | 420.63M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet272bn_cifar100-2007-5357e0df.tf2.h5.log)) |
| ResNet-542(BN) | 19.32 | 5,622,196 | 833.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet542bn_cifar100-1932-2db913a6.tf2.h5.log)) |
| ResNet-1001 | 19.79 | 10,351,732 | 1,536.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet1001_cifar100-1979-75c8acac.tf2.h5.log)) |
| ResNet-1202 | 21.56 | 19,429,876 | 2,857.17M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet1202_cifar100-2156-28fcf786.tf2.h5.log)) |
| PreResNet-20 | 30.22 | 278,132 | 41.28M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet20_cifar100-3022-447255f8.tf2.h5.log)) |
| PreResNet-56 | 25.05 | 861,428 | 127.04M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet56_cifar100-2505-180fc208.tf2.h5.log)) |
| PreResNet-110 | 22.67 | 1,736,372 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet110_cifar100-2267-ab677c09.tf2.h5.log)) |
| PreResNet-164(BN) | 20.18 | 1,726,388 | 255.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet164bn_cifar100-2018-c7649701.tf2.h5.log)) |
| PreResNet-272(BN) | 19.63 | 2,839,220 | 420.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet272bn_cifar100-1963-22e09198.tf2.h5.log)) |
| PreResNet-542(BN) | 18.71 | 5,621,300 | 833.66M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet542bn_cifar100-1871-703875c6.tf2.h5.log)) |
| PreResNet-1001 | 18.41 | 10,350,836 | 1,536.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet1001_cifar100-1841-7481e79c.tf2.h5.log)) |
| ResNeXt-20 (1x64d) | 21.97 | 3,538,852 | 538.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_1x64d_cifar100-2197-413945af.tf2.h5.log)) |
| ResNeXt-20 (2x32d) | 22.55 | 2,764,708 | 425.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_2x32d_cifar100-2255-bf34e56a.tf2.h5.log)) |
| ResNeXt-20 (4x16d) | 23.04 | 2,377,636 | 368.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_4x16d_cifar100-2304-fa8d4e06.tf2.h5.log)) |
| ResNeXt-20 (8x8d) | 22.82 | 2,184,100 | 340.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_8x8d_cifar100-2282-51922108.tf2.h5.log)) |
| ResNeXt-20 (16x4d) | 22.82 | 2,087,332 | 326.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_16x4d_cifar100-2282-e800aabb.tf2.h5.log)) |
| ResNeXt-20 (32x2d) | 21.73 | 2,038,948 | 319.12M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_32x2d_cifar100-2322-2def8cc2.tf2.h5.log)) |
| ResNeXt-20 (64x1d) | 23.53 | 2,014,756 | 315.58M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x1d_cifar100-2353-91695baa.tf2.h5.log)) |
| ResNeXt-20 (2x64d) | 20.60 | 6,290,852 | 988.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_2x64d_cifar100-2060-6eef33bc.tf2.h5.log)) |
| ResNeXt-20 (4x32d) | 21.31 | 4,742,564 | 761.66M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_4x32d_cifar100-2131-edabd5da.tf2.h5.log)) |
| ResNeXt-20 (8x16d) | 21.72 | 3,968,420 | 648.46M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_8x16d_cifar100-2172-3665fda7.tf2.h5.log)) |
| ResNeXt-20 (16x8d) | 21.73 | 3,581,348 | 591.86M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_16x8d_cifar100-2173-0a330298.tf2.h5.log)) |
| ResNeXt-20 (32x4d) | 22.13 | 3,387,812 | 563.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_32x4d_cifar100-2213-9508c15d.tf2.h5.log)) |
| ResNeXt-20 (64x2d) | 22.35 | 3,291,044 | 549.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x2d_cifar100-2235-e4a559cc.tf2.h5.log)) |
| ResNeXt-56 (1x64d) | 18.25 | 9,409,444 | 1,399.42M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_1x64d_cifar100-1825-72700951.tf2.h5.log)) |
| ResNeXt-56 (2x32d) | 17.86 | 7,087,012 | 1,059.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_2x32d_cifar100-1786-6639c30d.tf2.h5.log)) |
| ResNeXt-56 (4x16d) | 18.09 | 5,925,796 | 890.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_4x16d_cifar100-1809-61b41c3b.tf2.h5.log)) |
| ResNeXt-56 (8x8d) | 18.06 | 5,345,188 | 805.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_8x8d_cifar100-1806-f3f80382.tf2.h5.log)) |
| ResNeXt-56 (16x4d) | 18.24 | 5,054,884 | 762.65M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_16x4d_cifar100-1824-667ba183.tf2.h5.log)) |
| ResNeXt-56 (32x2d) | 18.60 | 4,909,732 | 741.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_32x2d_cifar100-1860-7a236896.tf2.h5.log)) |
| ResNeXt-56 (64x1d) | 18.16 | 4,837,156 | 730.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_64x1d_cifar100-1816-06c6c7a0.tf2.h5.log)) |
| ResNeXt-29 (32x4d) | 19.50 | 4,868,004 | 780.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext29_32x4d_cifar100-1950-e9979139.tf2.h5.log)) |
| ResNeXt-29 (16x64d) | 16.93 | 68,247,460 | 10,709.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext29_16x64d_cifar100-1693-2df09272.tf2.h5.log)) |
| ResNeXt-272 (1x64d) | 19.11 | 44,632,996 | 6,565.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext272_1x64d_cifar100-1911-e9275c94.tf2.h5.log)) |
| ResNeXt-272 (2x32d) | 18.34 | 33,020,836 | 4,867.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext272_2x32d_cifar100-1834-274ef607.tf2.h5.log)) |
| SE-ResNet-20 | 28.54 | 280,697 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet20_cifar100-2854-598b5858.tf2.h5.log)) |
| SE-ResNet-56 | 22.94 | 868,739 | 127.07M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet56_cifar100-2294-9c86ec99.tf2.h5.log)) |
| SE-ResNet-110 | 20.86 | 1,750,802 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet110_cifar100-2086-6435b022.tf2.h5.log)) |
| SE-ResNet-164(BN) | 19.95 | 1,929,388 | 256.57M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet164bn_cifar100-1995-121a777a.tf2.h5.log)) |
| SE-ResNet-272(BN) | 19.07 | 3,176,956 | 422.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet272bn_cifar100-1907-a29e50de.tf2.h5.log)) |
| SE-ResNet-542(BN) | 18.87 | 6,295,876 | 838.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet542bn_cifar100-1887-ddc4d5c8.tf2.h5.log)) |
| SE-PreResNet-20 | 28.31 | 280,409 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet20_cifar100-2831-e8dab8b8.tf2.h5.log)) |
| SE-PreResNet-56 | 23.05 | 868,451 | 127.21M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet56_cifar100-2305-aea4d90b.tf2.h5.log)) |
| SE-PreResNet-110 | 22.61 | 1,750,514 | 255.99M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet110_cifar100-2261-19a8d4a1.tf2.h5.log)) |
| SE-PreResNet-164(BN) | 20.05 | 1,928,012 | 256.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet164bn_cifar100-2005-9c3ed250.tf2.h5.log)) |
| SE-PreResNet-272(BN) | 19.13 | 3,175,580 | 422.47M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet272bn_cifar100-1913-eb75217f.tf2.h5.log)) |
| SE-PreResNet-542(BN) | 19.45 | 6,294,500 | 837.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet542bn_cifar100-1945-969d2bf0.tf2.h5.log)) |
| PyramidNet-110 (a=48) | 20.95 | 1,778,556 | 408.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a48_cifar100-2095-3490690a.tf2.h5.log)) |
| PyramidNet-110 (a=84) | 18.87 | 3,913,536 | 778.16M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a84_cifar100-1887-85789d68.tf2.h5.log)) |
| PyramidNet-110 (a=270) | 17.10 | 28,511,307 | 4,730.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a270_cifar100-1710-cc58021f.tf2.h5.log)) |
| PyramidNet-164 (a=270, BN) | 16.70 | 27,319,071 | 4,608.91M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet164_a270_bn_cifar100-1670-25ddf056.tf2.h5.log)) |
| PyramidNet-200 (a=240, BN) | 16.09 | 26,844,952 | 4,563.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet200_a240_bn_cifar100-1609-d2b16822.tf2.h5.log)) |
| PyramidNet-236 (a=220, BN) | 16.34 | 27,054,096 | 4,631.41M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet236_a220_bn_cifar100-1634-37d5b197.tf2.h5.log)) |
| PyramidNet-272 (a=200, BN) | 16.19 | 26,288,692 | 4,541.43M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet272_a200_bn_cifar100-1619-5c233384.tf2.h5.log)) |
| DenseNet-40 (k=12) | 24.90 | 622,360 | 210.82M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k12_cifar100-2490-ef38ff65.tf2.h5.log)) |
| DenseNet-BC-40 (k=12) | 28.41 | 188,092 | 74.90M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k12_bc_cifar100-2841-c7fbb0f4.tf2.h5.log)) |
| DenseNet-BC-40 (k=24) | 22.67 | 714,196 | 293.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k24_bc_cifar100-2267-b3878e82.tf2.h5.log)) |
| DenseNet-BC-40 (k=36) | 20.50 | 1,578,412 | 654.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k36_bc_cifar100-2050-045ae83a.tf2.h5.log)) |
| DenseNet-100 (k=12) | 19.65 | 4,129,600 | 1,353.62M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k12_cifar100-1965-4f0083d6.tf2.h5.log)) |
| DenseNet-100 (k=24) | 18.08 | 16,236,268 | 5,354.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k24_cifar100-1808-b0842c59.tf2.h5.log)) |
| DenseNet-BC-100 (k=12) | 21.19 | 800,032 | 298.48M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k12_bc_cifar100-2119-c1b857d5.tf2.h5.log)) |
| DenseNet-BC-250 (k=24) | 17.39 | 15,480,556 | 5,519.69M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet250_k24_bc_cifar100-1739-02d967b5.tf2.h5.log)) |

### SVHN

| Model | Error, % | Params | FLOPs/2 | Remarks |
| --- | ---: | ---: | ---: | --- |
| ResNet-20 | 3.43 | 272,474 | 41.29M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet20_svhn-0343-3480eec0.tf2.h5.log)) |
| ResNet-56 | 2.75 | 855,770 | 127.06M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet56_svhn-0275-5acc5537.tf2.h5.log)) |
| ResNet-110 | 2.45 | 1,730,714 | 255.70M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet110_svhn-0245-a07e849f.tf2.h5.log)) |
| ResNet-164(BN) | 2.42 | 1,704,154 | 255.31M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet164bn_svhn-0242-1bfa8083.tf2.h5.log)) |
| ResNet-272(BN) | 2.43 | 2,816,986 | 420.61M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet272bn_svhn-0243-e2a8e355.tf2.h5.log)) |
| ResNet-542(BN) | 2.34 | 5,599,066 | 833.87M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet542bn_svhn-0234-0d6759e7.tf2.h5.log)) |
| ResNet-1001 | 2.41 | 10,328,602 | 1,536.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.438/resnet1001_svhn-0241-c9a01550.tf2.h5.log)) |
| PreResNet-20 | 3.22 | 272,282 | 41.27M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet20_svhn-0322-6dcae612.tf2.h5.log)) |
| PreResNet-56 | 2.80 | 855,578 | 127.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet56_svhn-0280-6e074c73.tf2.h5.log)) |
| PreResNet-110 | 2.79 | 1,730,522 | 255.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet110_svhn-0279-226a0b34.tf2.h5.log)) |
| PreResNet-164(BN) | 2.58 | 1,703,258 | 255.08M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet164bn_svhn-0258-2307c36f.tf2.h5.log)) |
| PreResNet-272(BN) | 2.34 | 2,816,090 | 420.38M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet272bn_svhn-0234-3451d5fb.tf2.h5.log)) |
| PreResNet-542(BN) | 2.36 | 5,598,170 | 833.64M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.439/preresnet542bn_svhn-0236-5ca07592.tf2.h5.log)) |
| ResNeXt-20 (1x64d) | 2.98 | 3,446,602 | 538.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_1x64d_svhn-0298-105736c8.tf2.h5.log)) |
| ResNeXt-20 (2x32d) | 2.96 | 2,672,458 | 425.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_2x32d_svhn-0296-b61e1395.tf2.h5.log)) |
| ResNeXt-20 (4x16d) | 3.17 | 2,285,386 | 368.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_4x16d_svhn-0317-cab6d9fd.tf2.h5.log)) |
| ResNeXt-20 (8x8d) | 3.18 | 2,091,850 | 340.25M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_8x8d_svhn-0318-6ef55252.tf2.h5.log)) |
| ResNeXt-20 (16x4d) | 3.21 | 1,995,082 | 326.10M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_16x4d_svhn-0321-77a670a8.tf2.h5.log)) |
| ResNeXt-20 (32x2d) | 3.27 | 1,946,698 | 319.03M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_32x2d_svhn-0327-0c099194.tf2.h5.log)) |
| ResNeXt-20 (64x1d) | 3.42 | 1,922,506 | 315.49M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x1d_svhn-0342-a3bad459.tf2.h5.log)) |
| ResNeXt-20 (2x64d) | 2.83 | 6,198,602 | 987.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_2x64d_svhn-0283-dedfbac2.tf2.h5.log)) |
| ResNeXt-20 (4x32d) | 2.98 | 4,650,314 | 761.57M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_4x32d_svhn-0298-82b75cbb.tf2.h5.log)) |
| ResNeXt-20 (8x16d) | 3.01 | 3,876,170 | 648.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_8x16d_svhn-0301-d1a547e4.tf2.h5.log)) |
| ResNeXt-20 (16x8d) | 2.93 | 3,489,098 | 591.77M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_16x8d_svhn-0293-4ebac276.tf2.h5.log)) |
| ResNeXt-20 (32x4d) | 3.09 | 3,295,562 | 563.47M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_32x4d_svhn-0309-c8a843e1.tf2.h5.log)) |
| ResNeXt-20 (64x2d) | 3.14 | 3,198,794 | 549.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext20_64x2d_svhn-0314-c755e25d.tf2.h5.log)) |
| ResNeXt-56 (1x64d) | 2.42 | 9,317,194 | 1,399.33M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_1x64d_svhn-0242-dd7ac31e.tf2.h5.log)) |
| ResNeXt-56 (2x32d) | 2.46 | 6,994,762 | 1,059.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_2x32d_svhn-0246-61524d8a.tf2.h5.log)) |
| ResNeXt-56 (4x16d) | 2.44 | 5,833,546 | 889.91M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_4x16d_svhn-0244-b7ab2469.tf2.h5.log)) |
| ResNeXt-56 (8x8d) | 2.47 | 5,252,938 | 805.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_8x8d_svhn-0247-85692d77.tf2.h5.log)) |
| ResNeXt-56 (16x4d) | 2.56 | 4,962,634 | 762.56M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_16x4d_svhn-0256-86f327a9.tf2.h5.log)) |
| ResNeXt-56 (32x2d) | 2.53 | 4,817,482 | 741.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_32x2d_svhn-0253-b93a0535.tf2.h5.log)) |
| ResNeXt-56 (64x1d) | 2.55 | 4,744,906 | 730.72M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext56_64x1d_svhn-0255-9e9e3cc2.tf2.h5.log)) |
| ResNeXt-29 (32x4d) | 2.80 | 4,775,754 | 780.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext29_32x4d_svhn-0280-de6cba99.tf2.h5.log)) |
| ResNeXt-29 (16x64d) | 2.68 | 68,155,210 | 10,709.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext29_16x64d_svhn-0268-c929fada.tf2.h5.log)) |
| ResNeXt-272 (1x64d) | 2.34 | 44,540,746 | 6,565.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext272_1x64d_svhn-0234-4d348e9e.tf2.h5.log)) |
| ResNeXt-272 (2x32d) | 2.44 | 32,928,586 | 4,867.11M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.440/resnext272_2x32d_svhn-0244-f7923965.tf2.h5.log)) |
| SE-ResNet-20 | 3.23 | 274,847 | 41.34M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet20_svhn-0323-ef43ce80.tf2.h5.log)) |
| SE-ResNet-56 | 2.64 | 862,889 | 127.19M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet56_svhn-0264-a8fcc570.tf2.h5.log)) |
| SE-ResNet-110 | 2.35 | 1,744,952 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet110_svhn-0235-57751ac7.tf2.h5.log)) |
| SE-ResNet-164(BN) | 2.45 | 1,906,258 | 256.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet164bn_svhn-0245-a19e2e88.tf2.h5.log)) |
| SE-ResNet-272(BN) | 2.38 | 3,153,826 | 422.68M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet272bn_svhn-0238-918ee0de.tf2.h5.log)) |
| SE-ResNet-542(BN) | 2.26 | 6,272,746 | 838.01M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.442/seresnet542bn_svhn-0226-5ec784aa.tf2.h5.log)) |
| SE-PreResNet-20 | 3.24 | 274,559 | 41.35M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet20_svhn-0324-e7dbcc96.tf2.h5.log)) |
| SE-PreResNet-56 | 2.71 | 862,601 | 127.20M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet56_svhn-0271-ea024196.tf2.h5.log)) |
| SE-PreResNet-110 | 2.59 | 1,744,664 | 255.98M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet110_svhn-0259-6291c548.tf2.h5.log)) |
| SE-PreResNet-164(BN) | 2.56 | 1,904,882 | 256.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet164bn_svhn-0256-c8952322.tf2.h5.log)) |
| SE-PreResNet-272(BN) | 2.49 | 3,152,450 | 422.45M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet272bn_svhn-0249-0a778e9d.tf2.h5.log)) |
| SE-PreResNet-542(BN) | 2.47 | 6,271,370 | 837.78M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.443/sepreresnet542bn_svhn-0247-8e242736.tf2.h5.log)) |
| PyramidNet-110 (a=48) | 2.47 | 1,772,706 | 408.37M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a48_svhn-0247-15827390.tf2.h5.log)) |
| PyramidNet-110 (a=84) | 2.43 | 3,904,446 | 778.15M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a84_svhn-0243-aacb5f88.tf2.h5.log)) |
| PyramidNet-110 (a=270) | 2.38 | 28,485,477 | 4,730.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet110_a270_svhn-0238-b8742320.tf2.h5.log)) |
| PyramidNet-164 (a=270, BN) | 2.34 | 27,216,021 | 4,608.81M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet164_a270_bn_svhn-0234-94bb4029.tf2.h5.log)) |
| PyramidNet-200 (a=240, BN) | 2.32 | 26,752,702 | 4,563.40M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet200_a240_bn_svhn-0232-77f2380c.tf2.h5.log)) |
| PyramidNet-236 (a=220, BN) | 2.35 | 26,969,046 | 4,631.32M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet236_a220_bn_svhn-0235-6a9a8b0a.tf2.h5.log)) |
| PyramidNet-272 (a=200, BN) | 2.40 | 26,210,842 | 4,541.36M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.444/pyramidnet272_a200_bn_svhn-0240-0a389e2f.tf2.h5.log)) |
| DenseNet-40 (k=12) | 3.05 | 599,050 | 210.80M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k12_svhn-0305-7d5860ae.tf2.h5.log)) |
| DenseNet-BC-40 (k=12) | 3.20 | 176,122 | 74.89M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k12_bc_svhn-0320-77fd3ddf.tf2.h5.log)) |
| DenseNet-BC-40 (k=24) | 2.90 | 690,346 | 293.09M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k24_bc_svhn-0290-b8a231f7.tf2.h5.log)) |
| DenseNet-BC-40 (k=36) | 2.60 | 1,542,682 | 654.60M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet40_k36_bc_svhn-0260-a176dcf1.tf2.h5.log)) |
| DenseNet-100 (k=12) | 2.60 | 4,068,490 | 1,353.55M | Converted from GL model ([log](https://github.com/osmr/imgclsmob/releases/download/v0.0.445/densenet100_k12_svhn-0260-e810c380.tf2.h5.log)) |

[dmlc/gluon-cv]: https://github.com/dmlc/gluon-cv
[tornadomeet/ResNet]: https://github.com/tornadomeet/ResNet
[Cadene/pretrained...pytorch]: https://github.com/Cadene/pretrained-models.pytorch
[tensorpack/tensorpack]: https://github.com/tensorpack/tensorpack
[clavichord93/MENet]: https://github.com/clavichord93/MENet
[zeusees/Mnasnet...Model]: https://github.com/zeusees/Mnasnet-Pretrained-Model
[soeaver/mxnet-model]: https://github.com/soeaver/mxnet-model
[rwightman/pyt...models]: https://github.com/rwightman/pytorch-image-models
[soeaver/AirNet-PyTorch]: https://github.com/soeaver/AirNet-PyTorch
[dyhan0920/Pyramid...PyTorch]: https://github.com/dyhan0920/PyramidNet-PyTorch
[szagoruyko/functional-zoo]: https://github.com/szagoruyko/functional-zoo
[Jongchan/attention-module]: https://github.com/Jongchan/attention-module
[wielandbrendel/bag...models]: https://github.com/wielandbrendel/bag-of-local-features-models
[fyu/drn]: https://github.com/fyu/drn
[ucbdrive/dla]: https://github.com/ucbdrive/dla
[XingangPan/IBN-Net]: https://github.com/XingangPan/IBN-Net
[HRNet/HRNet...ation]: https://github.com/HRNet/HRNet-Image-Classification
[stigma0617/VoVNet.pytorch]: https://github.com/stigma0617/VoVNet.pytorch
[PingoLH/Pytorch-HarDNet]: https://github.com/PingoLH/Pytorch-HarDNet