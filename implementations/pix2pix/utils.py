
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from .model import Generator, Discriminator, weights_init_normal

from dataset import DanbooruAutoPair
from utils import Status, save_args, add_args
from nnutils import get_device
from nnutils.loss import LSGANLoss

def train(
    max_iters, dataset,
    G, optimizer_G,
    D, optimizer_D,
    pixelwise_criterion, pixelwise_gamma,
    device, save_interval
):

    status = Status(max_iters)
    loss   = LSGANLoss()

    while status.batches_done < max_iters:
        for index, (target, pair) in enumerate(dataset, 1):
            # target image
            target = target.to(device)
            # pair image
            pair = pair.to(device)
            print(target.size(), pair.size())

            # generate image
            fake = G(pair)

            '''
            Train Discriminator
            '''

            # real
            real_prob = D(torch.cat([target, pair], dim=1))
            # fake
            fake_prob = D(torch.cat([target, fake.detach()], dim=1))

            D_loss = loss.d_loss(real_prob, fake_prob)

            # optimize
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            '''
            Train Generator
            '''

            # train to fool D
            fake_prob= D(torch.cat([target, fake], dim=1))

            # loss
            adv_loss = loss.g_loss(fake_prob)
            g_pixelwise_loss = pixelwise_criterion(fake, target)
            G_loss = adv_loss + g_pixelwise_loss * pixelwise_gamma

            # optimize
            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

            if status.batches_done % save_interval == 0:
                save_image(
                    fake, f'implementations/pix2pix/result/{status.batches_done}.jpg',
                    normalize=True, value_range=(-1, 1))

            status.update(d=D_loss.item(), g=G_loss.item())
            if status.batches_done == max_iters:
                break

    status.plot_loss()

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

def main(parser):

    parser = add_args(parser,
        dict(
            lr              = [0.0005, 'learning rate'],
            betas           = [[0.5, 0.999], 'betas'],
            pixelwise_gamma = [100., 'gamma for l1 loss']))
    parser.set_defaults(image_size=256)
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)

    pair_transform = Mozaic(linear_scale=2)
    dataset = DanbooruAutoPair.asloader(
        args.batch_size, (args.image_size, pair_transform, args.num_images),
        pin_memory=not args.disable_gpu)

    G = Generator()
    D = Discriminator()
    G.apply(weights_init_normal)
    D.apply(weights_init_normal)
    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    pixelwise_criterion = nn.L1Loss()

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    train(
        args.max_iters, dataset,
        G, optimizer_G, D, optimizer_D,
        pixelwise_criterion, args.pixelwise_gamma,
        device, args.save)
