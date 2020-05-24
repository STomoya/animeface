from general import *
from GAN import main as gan_main
from DCGAN import main as dcgan_main
from cGAN import main as cgan_main
from ACGAN import main as acgan_main
from pixelshuffle import main as pixelshuffle_main

def main():
    image_size = 128
    batch_size = 64

    # dataset = AnimeFaceDataset(image_size=image_size)
    dataset = OneHotLabeledAnimeFaceDataset(image_size=image_size)
    dataset = to_loader(dataset=dataset, batch_size=batch_size)

    pixelshuffle_main(
        dataset=dataset,
        image_size=image_size
    )

if __name__ == "__main__":
    main()