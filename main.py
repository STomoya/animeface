from implementations.general import *
from implementations.GAN import main as gan_main
from implementations.DCGAN import main as dcgan_main
from implementations.cGAN import main as cgan_main
from implementations.ACGAN import main as acgan_main
from implementations.pixelshuffle import main as pixelshuffle_main
from implementations.WGAN import main as wgan_main
from implementations.WGAN_gp import main as wgan_gp_main

from implementations.PGGAN import main as pggan_main
from implementations.StyleGAN import main as stylegan_main
from implementations.StyleGAN2 import main as stylegan2_main
from implementations.HoloGAN import main as hologan_main

from implementations.pix2pix import main as pix2pix_main
from implementations.UGATIT import main as ugatit_main

from implementations.SinGAN import main as singan_main

from implementations.DiffAugment import main as da_main

def main():
    # image_size = 128
    # batch_size = 64

    # dataset = AnimeFaceDataset(image_size=image_size)
    # dataset = OneHotLabeledAnimeFaceDataset(image_size=image_size)
    # dataset = to_loader(dataset=dataset, batch_size=batch_size)

    # wgan_gp_main(
    #     dataset=dataset,
    #     image_size=image_size
    # )

    # ugatit_main(
    #     GeneratePairImageDanbooruDataset,
    #     to_loader
    # )

    hologan_main(
        AnimeFaceDataset,
        to_loader,
        DiffAugment
    )

    # singan_main()

if __name__ == "__main__":
    main()