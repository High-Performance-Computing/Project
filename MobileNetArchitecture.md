# MobileNet and MobileNetV2 Architectures

## MobileNet

<p align="justify"> We first examine the paper [MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications](https://arxiv.org/abs/1704.04861). The objectives of the author is to make neural networks much lighter, with an eye on deployement on mobiles and embedded vision applications, without compromising on accuracy. </p>

### Depthwise separable convolutions
 
<p align="justify"> The key idea behind MobileNets are depthwise separable convolutions. </p>

![](RegularConvolution.png)

<p align="justify"> In a regular convolution, we apply a filter a D<sub>K</sub> by D<sub>K</sub> by M filter. So we look for relationships within each channel (D<sub>K</sub> by D<sub>K</sub>) and between channels. </p>

<p align="justify"> The goal of depthwise seprable convolutions is to break down this process into two parts: first, we look for relationships within each channel, and then between different channels. This is illustrated below in figures b) and c). We first take D<sub>K</sub> by D<sub>K</sub> filters and apply them one each to a channel. We thus have M such filers, called depthwise convolutions. In image c), we can see the pointwise convolutions: those are 1 by 1 by M convolutions. </p>


![](Mobilenet.png)

<p align="justify"> The cost of a regular convolution is  D<sub>K</sub>  D<sub>K</sub> M. If we do N such convolutions, the cost is D<sub>K</sub>  D<sub>K</sub> M N. Finally, if we apply this to our block of size D<sub>F</sub> by   D<sub>F</sub>, with stride one, we get a cost of D<sub>K</sub>  D<sub>K</sub> M N  D<sub>F</sub>  D<sub>F</sub>. </p>


