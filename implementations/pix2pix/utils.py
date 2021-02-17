
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image
import numpy as np

from .model import Generator, Discriminator, weights_init_normal

from ..general import GeneratePairImageDanbooruDataset, to_loader, save_args

def train(
    epochs,
    dataset,
    G,
    optimizer_G,
    D,
    optimizer_D,
    validity_criterion,
    pixelwise_criterion,
    pixelwise_gamma,
    device,
    verbose_interval,
    save_interval
):

    patch = (1, 16, 16)
    losses = {
        'g' : [],
        'd' : []
    }
    
    for epoch in range(epochs):
        for index, (target, pair) in enumerate(dataset, 1):
            # label smoothing
            gt   = torch.from_numpy((1.0 - 0.7) * np.random.randn(target.size(0), *patch) + 0.7)
            gt   = gt.type(torch.FloatTensor).to(device)
            fake = torch.from_numpy((0.3 - 0.0) * np.random.randn(target.size(0), *patch) + 0.0)
            fake = fake.type(torch.FloatTensor).to(device)

            # target image
            target = target.type(torch.FloatTensor).to(device)
            # pair image
            pair = pair.type(torch.FloatTensor).to(device)


            # generate image
            gen_image = G(pair)


            '''
            Train Discriminator
            '''

            # real
            pred_real = D(torch.cat([target, pair], dim=1))
            d_real_loss = validity_criterion(pred_real, gt)

            # fake
            pred_fake = D(torch.cat([target, gen_image.detach()], dim=1))
            d_fake_loss = validity_criterion(pred_fake, fake)

            d_loss = (d_real_loss + d_fake_loss) / 2

            # optimize
            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()


            '''
            Train Generator
            '''


            # train to fool D
            pred_fake = D(torch.cat([target, gen_image], dim=1))
            g_validity_loss = validity_criterion(pred_fake, gt)
            g_pixelwise_loss = pixelwise_criterion(gen_image, target)
            g_loss = g_validity_loss + g_pixelwise_loss * pixelwise_gamma


            # optimize
            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()


            losses['g'].append(g_loss.item())
            losses['d'].append(d_loss.item())

            batches_done = epoch * len(dataset) + index

            if batches_done % verbose_interval == 0:
                print(
                    "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f]"
                    % (epoch, epochs, index, len(dataset), d_loss.item(), g_loss.item())
                )
            

            if batches_done % save_interval == 0:
                num_images = 32
                images = grid(pair, gen_image, target, num_images)
                save_image(images, "implementations/pix2pix/result/%d.png" % batches_done, nrow=4*3, normalize=True)

    return losses

def grid(pair, gen_image, target, num_images):
    image_shape = target.size()[1:]

    pair_tuple   = torch.unbind(pair)
    gen_tuple    = torch.unbind(gen_image)
    target_tuple = torch.unbind(target)

    images = []
    for pair, gen_image, target in zip(pair_tuple, gen_tuple, target_tuple):
        images = images + [pair.view(1, *image_shape), gen_image.view(1, *image_shape), target.view(1, *image_shape)]

        if len(images) >= num_images*3:
            break
    return torch.cat(images)

from PIL import ImageFilter

class Mozaic(object):
    def __init__(self, linear_scale):
        self.linear_scale = linear_scale

    def __call__(self, sample):
        return sample.resize([x // self.linear_scale for x in sample.size]).resize(sample.size)

class SoftMozaic(object):
    def __init__(self, linear_scale, radius=2):
        self.linear_scale = linear_scale
        self.radius = radius

    def __call__(self, sample):
        sample = sample.filter(ImageFilter.GaussianBlur(self.radius))
        return sample.resize([x // self.linear_scale for x in sample.size]).resize(sample.size)

class GaussianBlur(object):
    def __init__(self, radius):
        self.radius = radius

    def __call__(self, sample):
        return sample.filter(ImageFilter.GaussianBlur(self.radius))

def add_arguments(parser):
    parser.add_argument('--epochs', default=150, type=int, help='epochs to train')
    parser.add_argument('--lr', default=0.0005, type=float, help='learning rate')
    parser.add_argument('--beta1', default=0.5, type=float, help='beta1')
    parser.add_argument('--beta2', default=0.999, type=float, help='beta2')
    parser.add_argument('--pixelwise-gamma', default=100, type=float, help='gamma for l1 loss')
    return parser

def main(parser):

    parser = add_arguments(parser)
    args = parser.parse_args()
    save_args(args)

    betas = (args.beta1, args.beta2)

    pair_transform = Mozaic(linear_scale=2)
    dataset = GeneratePairImageDanbooruDataset(pair_transform=pair_transform, image_size=args.image_size)
    dataset = to_loader(dataset, args.batch_size)

    G = Generator()
    D = Discriminator()

    G.apply(weights_init_normal)
    D.apply(weights_init_normal)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=betas)

    validity_criterion = nn.MSELoss()
    pixelwise_criterion = nn.L1Loss()

    losses = train(
        epochs=args.epochs,
        dataset=dataset,
        G=G,
        optimizer_G=optimizer_G,
        D=D,
        optimizer_D=optimizer_D,
        validity_criterion=validity_criterion,
        pixelwise_criterion=pixelwise_criterion,
        pixelwise_gamma=args.pixelwise_gamma,
        device=device,
        verbose_interval=1000,
        save_interval=1000
    )
