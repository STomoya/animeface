
'''GANs'''
# adversarial loss
from implementations.WGAN import main as wgan_main
from implementations.WGAN_gp import main as wgan_gp_main

# generative models G(z)
from implementations.GAN import main as gan_main
from implementations.DCGAN import main as dcgan_main
from implementations.PGGAN import main as pggan_main
from implementations.StyleGAN import main as stylegan_main
from implementations.StyleGAN2 import main as stylegan2_main
from implementations.HoloGAN import main as hologan_main
from implementations.BigGAN import main as biggan_main

# image manipulation
from implementations.SinGAN import main as singan_main

# conditional GANs G(z, t)
from implementations.cGAN import main as cgan_main
from implementations.ACGAN import main as acgan_main

# image-to-image translation
from implementations.pix2pix import main as pix2pix_main
from implementations.pix2pixHD import main as pix2pixhd_main
from implementations.UGATIT import main as ugatit_main

# optimizer
from implementations.AdaBelief import main as adabelief_main
# augmentation
from implementations.DiffAugment import main as da_main
# upsampling
from implementations.pixelshuffle import main as pixelshuffle_main

'''Auto Encoders'''
from implementations.AE import main as ae_main
from implementations.VAE import main as vae_main

def main():
    vae_main()

if __name__ == "__main__":
    main()