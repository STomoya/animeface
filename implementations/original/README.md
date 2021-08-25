
# original

Original models.  
(They might exist and I just don't know about it.)

## Models

- EDCNN

    Using the edge enhancement module in EDCNN for richer edge info for image-to-image translation with anime images.

    [Method overview](EDCNN/image/edge-enhance-overview.png)

    - thoughts

        Same results can be obtained without the use of this module.
        It maps colors that are not contained in the original image.
        Maybe there are more suitable tasks than colorization. (Originally is for denoising)

- SEBigGAN

    Use Squeeze and Excitation Network instead of Self-Attention layer in BigGAN.

- StyleGAN2

    Playing with StyleGAN2.

    - Train with anime+photo. (Randomly mixed and separate)
    - etc...

## Author

[Tomoya Sawada](https://github.com/STomoya)
