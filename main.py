from general import *
from GAN import main as gan_main
from DCGAN import main as dcgan_main
from cGAN import main as cgan_main
from ACGAN import main as acgan_main
from pixelshuffle import main as pixelshuffle_main
from WGAN import main as wgan_main
from WGAN_gp import main as wgan_gp_main

from PGGAN import main as pggan_main
from StyleGAN import main as stylegan_main
from StyleGAN2 import main as stylegan2_main

from DiffAugment import main as da_main

from pix2pix import main as pix2pix_main
from UGATIT import main as ugatit_main

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

    stylegan2_main(
        AnimeFaceDataset,
        to_loader,
        DiffAugment
    )

if __name__ == "__main__":
    main()