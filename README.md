
<p align="center">
    <img alt="StyleGAN2 result" src="https://raw.githubusercontent.com/STomoya/animeface/master/implementations/StyleGAN2/result/118000.png">
</p>

# animeface

deep models for anime images.

## Datasets

- [anime-face-dataset](https://www.kaggle.com/splcher/animefacedataset)  
    Anime faces collected from [Getchu.com](http://www.getchu.com/).  
    Based on [Mckinsey666](https://github.com/Mckinsey666/Anime-Face-Dataset)'s dataset.  
    63.6K images.
- [Tagged Anime Illustrations](https://www.kaggle.com/mylesoneill/tagged-anime-illustrations)  
    A subset of the [Danbooru2017](https://www.gwern.net/Danbooru2017), and the [moeimouto face dataset](http://www.nurs.or.jp/~nagadomi/animeface-character-dataset/).  
    337K Danbooru images, 17.4K moeimouto face images.
- [Danbooru2019 Portraits](https://www.gwern.net/Crops#danbooru2019-portraits) [1]  
    Portraits of anime characters collected from [Danbooru2019](https://www.gwern.net/Danbooru2019).  
    302K portraits.

## Models

"code" is indicated when only an official implementation exists.

### Generative Adversarial Networks (GANs)

|Method|paper|code|
|------|-----|----|
|Augmented-Self Reference and Dense Semantic Correspondence (SCFT)|[arxiv](https://arxiv.org/abs/2005.05207)|-|
|auxiliary classifier GAN (ACGAN)|[arxiv](https://arxiv.org/abs/1610.09585)|-|
|big GAN (BigGAN)|[arxiv](https://arxiv.org/abs/1809.11096)|[github](https://github.com/ajbrock/BigGAN-PyTorch)|
|conditional GAN (cGAN)|[arxiv](https://arxiv.org/abs/1411.1784)|-|
|Contrastive Discriminator (ContraD)|[openreview](https://openreview.net/forum?id=eo6U4CAwVmg)|[github](https://github.com/jh-jeong/ContraD)
|deep convolutional GAN (DCGAN)|[arxiv](https://arxiv.org/abs/1511.06434)|-|
|deep regret analytic GAN (DRAGAN)|[arxiv](https://arxiv.org/abs/1705.07215)|[github](https://github.com/kodalinaveen3/DRAGAN)|
|Enhanced super resolution GAN (ESRGAN)|[arxiv](https://arxiv.org/abs/1809.00219)|[github](https://github.com/xinntao/ESRGAN)|
|GANILLA|[arxiv](https://arxiv.org/abs/2002.05638)|[github](https://github.com/giddyyupp/ganilla)|
|generative adversarial networks (GAN)|[arxiv](https://arxiv.org/abs/1406.2661)|-|
|Image-to-image Translation via Hierarchical Style Disentanglement (HiSD)|[arxiv](https://arxiv.org/abs/2103.01456)|[github](https://github.com/imlixinyang/HiSD)|
|Hologram(?) GAN (HoloGAN)|[arxiv](https://arxiv.org/abs/1904.01326)|[github](https://github.com/thunguyenphuoc/HoloGAN)
|progressive growing of GANs (PGGAN)|[arxiv](https://arxiv.org/abs/1710.10196)|[github](https://github.com/tkarras/progressive_growing_of_gans)|
|pix2ix|[arxiv](https://arxiv.org/abs/1703.10593)|[github](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix)|
|pix2pix HD|[arxiv](https://arxiv.org/abs/1711.11585)|[github](https://github.com/NVIDIA/pix2pixHD)|
|single GAN (SinGAN)|[arxiv](https://arxiv.org/abs/1905.01164)|[github](https://github.com/tamarott/SinGAN)|
|Spatially-Adaptive Normalization (SPADE)|[arxiv](https://arxiv.org/abs/1903.07291)|[github](https://github.com/NVlabs/SPADE)|
|style-based GAN (StyleGAN)|[arxiv](https://arxiv.org/abs/1812.04948)|[github](https://github.com/NVlabs/stylegan)|
|style-based GAN 2 (StyleGAN2)|[arxiv](https://arxiv.org/abs/1912.04958)|[github](https://github.com/NVlabs/stylegan2)|
|Super resolution GAN (SRGAN)|[arxiv](https://arxiv.org/abs/1609.04802)|-|
|Transformer-based GAN (TransGAN)|[arxiv](https://arxiv.org/abs/2102.07074)|[github](https://github.com/VITA-Group/TransGAN)|
|unsupervised GAN with adaptive layer-instance normalization (UGATIT)|[arxiv](https://arxiv.org/abs/1907.10830)|[github](https://github.com/taki0112/UGATIT)|
|Wasserstein GAN (WGAN)|[arxiv](https://arxiv.org/abs/1701.07875)|-|
|WGAN with gradient penalty (WGAN_gp)|[arxiv](https://arxiv.org/abs/1704.00028)|-|
|zero-centered gradient penalty|[arxiv](https://arxiv.org/abs/1705.09367)|-|
|simplified zero-centered gradient penality|[arxiv](https://arxiv.org/abs/1801.04406)|[github](https://github.com/LMescheder/GAN_stability)|

### Auto Encoders

|Method|paper|code|
|------|-----|----|
|Auto Encoder (AE)|[tronto](https://www.cs.toronto.edu/~hinton/science.pdf)|-|
|Variational Auto Encoder (VAE)|[arxiv](https://arxiv.org/abs/1312.6114)|-|

### Other

|Method|paper|code|
|------|-----|----|
|AdaBelief optimizer|[arxiv](https://arxiv.org/abs/2010.07468)|[github](https://github.com/juntang-zhuang/Adabelief-Optimizer)|
|Adaptive Discriminator Augmentation (ADA)|[arxiv](https://arxiv.org/abs/2006.06676)|[github](https://github.com/NVlabs/stylegan2-ada)|
|differentiable augmentation (DiffAugment)|[arxiv](https://arxiv.org/abs/2006.10738)|[github](https://github.com/mit-han-lab/data-efficient-gans)|
|pixel shuffle|[arxiv](https://arxiv.org/abs/1609.05158)|-|

## Weights

See `weights.md`

## Reference

```
[1] Gwern Branwen, Anonymous, & The Danbooru Community;
    “Danbooru2019 Portraits: A Large-Scale Anime Head Illustration Dataset”,
    2019-03-12. Web. Accessed 2020/09/17,
    https://www.gwern.net/Crops#danbooru2019-portraits
```

## Author

[Tomoya Sawada](https://github.com/STomoya)
