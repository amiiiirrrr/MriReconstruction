
# Compressed Sensing MRI Reconstruction Using Improved U-net based on Deep Generative Adversarial Networks

This is the official implementation code for [Compressed Sensing MRI Reconstruction Using Improved U-net based on Deep Generative Adversarial Networks](https://ieeexplore.ieee.org/document/9738554) published in International Conference on Machine Vision and Image Processing (MVIP) (2022).

If you use this code for your research, please cite our paper.

@inproceedings{mousavi2022compressed,
  title={Compressed Sensing MRI Reconstruction Using Improved U-net based on Deep Generative Adversarial Networks},
  author={Mousavi, Seyed Amir and Ahmadzadeh, Mohammad Reza and Yazdian, Ehsan},
  booktitle={2022 International Conference on Machine Vision and Image Processing (MVIP)},
  pages={1--8},
  year={2022},
  organization={IEEE}
}

If you have any questions about this code, please feel free to contact Seyed Amir Mousavi (azmusavi19@gmail.com).

## Abstract
Magnetic resonance imaging (MRI) as a non-invasive imaging is able to produces three dimensional detailed anatomical images without the use of damaging radiation and excellent visualization of both anatomical structure and physiological function. MRI is a time-consuming imaging technique and image quality may be reduced due to spontaneous or non-spontaneous movements of patient. Compressive Sensing MRI (CS-MRI) violates Nyquist-Shannon sampling rate and utilizes the sparsity of MR images to reconstruct MR images with under-sampled k-space data. prior works in CS-MRI approaches have employed orthogonal transforms such as wavelets and curvelets CS-MRI methods are based on constant transform bases or shallow dictionaries, which limits modeling capacity in this research, a novel method based on very deep convolutional neural networks (CNNs) for reconstruction MR images is proposed using Generative Adversarial Networks (GANs). in this model, Generative and Discriminator networks designed with improved Resnet architecture. Generative network is U-net based in which used from improved Resnet blocks, has led to reduction in aliasing artifacts, more accurate reconstruction of edges and better reconstruction of tissues. to achieve better reconstruction adversarial loss, pixel-wise cost and perceptual loss (pretrained deep VGG network) are combined. with assessment using various MRI databases such as brain, cardiac and prostate. the proposed method leads to better reconstruction in detail of image.

Generator model with Encoder and Decoder blocks:

<img src=https://user-images.githubusercontent.com/28767607/130682538-7136d817-d017-419e-bdd6-a3b8afe0d138.PNG width="500" height="300">

Encoder blocks in Generator:

<img src=https://user-images.githubusercontent.com/28767607/130683161-2c6665f7-fa16-4f33-a484-c6f14491184d.PNG width="300" height="200">

Decoder blocks in Generator:

<img src=https://user-images.githubusercontent.com/28767607/130683286-35771900-e190-47dc-a158-978124593a83.PNG width="300" height="200">

Discriminator model with Decoder blocks:

<img src=https://user-images.githubusercontent.com/28767607/130683406-dc3b2fc1-d7a7-4be4-bbc0-2e7a2ba881a6.PNG width="300" height="200">

Decoder blocks in Discriminator:

<img src=https://user-images.githubusercontent.com/28767607/130683627-ce08ccbc-b827-44f4-92d8-05d531f01f3c.PNG width="300" height="200">
