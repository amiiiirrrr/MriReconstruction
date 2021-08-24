
👋 Deep Generative Adversarial Networks(GANs) for Compressive Sensing MRI Reconstruction with improving U-net


Magnetic resonance imaging (MRI) as a non-invasive imaging is able to produces three dimensional detailed anatomical images without the use of damaging radiation and excellent visualization of both anatomical structure and physiological function. MRI is a time-consuming imaging technique and image quality may be reduced due to spontaneous or non-spontaneous movements of patient. Compressive Sensing MRI (CS-MRI) violates Nyquist-Shannon sampling rate and utilizes the sparsity of MR images to reconstruct MR images with under-sampled k-space data. prior works in CS-MRI approaches have employed orthogonal transforms such as wavelets and curvelets CS-MRI methods are based on constant transform bases or shallow dictionaries, which limits modeling capacity in this research, a novel method based on very deep convolutional neural networks (CNNs) for reconstruction MR images is proposed using Generative Adversarial Networks (GANs). in this model, Generative and Discriminator networks designed with improved Resnet architecture. Generative network is U-net based in which used from improved Resnet blocks, has led to reduction in aliasing artifacts, more accurate reconstruction of edges and better reconstruction of tissues. to achieve better reconstruction adversarial loss, pixel-wise cost and perceptual loss (pretrained deep VGG network) are combined. with assessment using various MRI databases such as brain, cardiac and prostate. the proposed method leads to better reconstruction in detail of image.


<img src=https://user-images.githubusercontent.com/28767607/130682538-7136d817-d017-419e-bdd6-a3b8afe0d138.PNG width="50" height="50">

