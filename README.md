<div align="center">  

## Multichannel input pixelwise regression 3D U-Nets for medical image estimation with 3 applications in brain MRI

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.4679670.svg)](https://doi.org/10.5281/zenodo.4679670)

</div>

## Description
Pytorch implementation of Multichannel input pixelwise regression 3D U-Nets.

## Abstract
The U-Net is a robust general-purpose deep learning architecture
  designed for semantic segmentation of medical images, and has
  been extended to 3D for volumetric applications such as
  magnetic resonance imaging (MRI) of the human brain. An
  adaptation of the U-Net to output pixelwise regression
  values, instead of class labels, based on multichannel input data
  has been developed in the remote sensing satellite imaging
  research domain. The pixelwise regression U-Net has only received
  limited consideration as a deep learning architecture in medical
  imaging for the image estimation/synthesis problem, and the
  limited work so far did not consider the application of 3D
  multichannel inputs. In this paper, we propose the use of the
  multichannel input pixelwise regression 3D U-Net (rUNet) for
  estimation of medical images. Our findings demonstrate that this
  approach is robust and versatile and can be applied to predicting
  a pending MRI examination of patients with Alzheimer's disease
  based on previous rounds of imaging, can perform medical image
  reconstruction (parametric mapping) in diffusion MRI, and can be
  applied to the estimation of one type of MRI examination from a
  collection of other types. Results demonstrate that the rUNet
  represents a single deep learning architecture capable of solving
  a variety of image estimation problems.

## Model
We use a 5-level 3D U-Net architecture, with Leaky ReLU activation 
($\alpha = 0.2$), learning rate ($\alpha = 10 ^ {-5}$), Adam 
optimizer, mean average error (MAE) loss function, z-score 
intensity normalization and co-registered volumes resized 
to 128x128x128 for each tasks. Batch size was 3 in application 1, 
and 1 in applications 2 and 3.

## Results
<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="这里输入图片地址">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Qualitative Results</div>
</center>

<center>
    <img style="border-radius: 0.3125em;
    box-shadow: 0 2px 4px 0 rgba(34,36,38,.12),0 2px 10px 0 rgba(34,36,38,.08);" 
    src="这里输入图片地址">
    <br>
    <div style="color:orange; border-bottom: 1px solid #d9d9d9;
    display: inline-block;
    color: #999;
    padding: 2px;">Quantitative Results</div>
</center>