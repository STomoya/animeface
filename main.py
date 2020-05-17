from general import *
from GAN import main as gan_main

def main():
    image_size = 100
    batch_size = 64

    dataset = AnimeFaceDataset(image_size=image_size)
    dataset = to_loader(dataset=dataset, batch_size=batch_size)

    gan_main(
        dataset=dataset,
        image_size=image_size
    )

if __name__ == "__main__":
    main()