
from functools import partial

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision.utils import save_image

from dataset import AnimeFace
from utils import Status, save_args, add_args
from nnutils import get_device, sample_unoise
from nnutils.loss import GANLoss
from thirdparty.diffaugment import DiffAugment

from .model import Generator, Discriminator

def gen_theta(
    num_gen, minmax_angles=[0, 0, 220, 320, 0, 0],
    random = True
):
    '''
    generate a batch of theta tensor.

    code from : christopher-beckham/hologan-pytorch
        URL     : https://github.com/christopher-beckham/hologan-pytorch/blob/master/hologan.py
        LICENSE : https://github.com/christopher-beckham/hologan-pytorch/blob/master/LICENSE
    modified by STomoya (https://github.com/STomoya)
    '''
    def radian(deg):
        return deg * (np.pi / 180)
    angles = {
        'x_min' : radian(minmax_angles[0]),
        'x_max' : radian(minmax_angles[1]),
        'y_min' : radian(minmax_angles[2]),
        'y_max' : radian(minmax_angles[3]),
        'z_min' : radian(minmax_angles[4]),
        'z_max' : radian(minmax_angles[5])
    }

    samples = []
    if random:
        # generate random angles (radian)
        for _ in range(num_gen):
            samples.append(
                [
                    np.random.uniform(angles['x_min'], angles['x_max']),
                    np.random.uniform(angles['y_min'], angles['y_max']),
                    np.random.uniform(angles['z_min'], angles['z_max'])
                ]
            )
    else:
        # generate samples from min to max
        x_tic = (angles['x_max'] - angles['x_min']) / (num_gen - 1)
        y_tic = (angles['y_max'] - angles['y_min']) / (num_gen - 1)
        z_tic = (angles['z_max'] - angles['z_min']) / (num_gen - 1)
        for times in range(num_gen):
            samples.append(
                [
                    angles['x_min'] + x_tic * times,
                    angles['y_min'] + y_tic * times,
                    angles['z_min'] + z_tic * times
                ]
            )
    samples = np.array(samples)

    # funcs for converting to rotation matrix
    def rotation_matrix_x(angle):
        matrix = np.zeros((3, 3)).astype(np.float32)
        matrix[0, 0] =   1.
        matrix[1, 1] =   np.cos(angle)
        matrix[1, 2] = - np.sin(angle)
        matrix[2, 1] =   np.sin(angle)
        matrix[2, 2] =   np.cos(angle)
        return matrix
    def rotation_matrix_y(angle):
        matrix = np.zeros((3, 3)).astype(np.float32)
        matrix[0, 0] =   np.cos(angle)
        matrix[0, 2] =   np.sin(angle)
        matrix[1, 1] =   1.
        matrix[2, 0] = - np.sin(angle)
        matrix[2, 2] =   np.cos(angle)
        return matrix
    def rotation_matrix_z(angle):
        matrix = np.zeros((3, 3)).astype(np.float32)
        matrix[0, 0] =   np.cos(angle)
        matrix[0, 1] = - np.sin(angle)
        matrix[1, 0] =   np.sin(angle)
        matrix[1, 1] =   np.cos(angle)
        matrix[2, 2] =   1.
        return matrix
    def pad_matrix(matrix):
        # pad 0 to fit size Bx3x4. (required by torch.nn.functional.affine_grid() for 5D tensor.)
        return np.hstack([matrix, np.zeros((3, 1))])

    # calc total rotation matrix
    samples_x = samples[:, 0]
    samples_y = samples[:, 1]
    samples_z = samples[:, 2]
    theta = np.zeros((num_gen, 3, 4))
    for i in range(num_gen):
        theta[i] = pad_matrix(np.dot(np.dot(rotation_matrix_z(samples_z[i]), rotation_matrix_y(samples_y[i])), rotation_matrix_x(samples_x[i])))

    theta = torch.from_numpy(theta).float()
    return theta

def train(
    max_iters, batch_size, noise_channels, eval_size,
    dataset, augment,
    G, D, optimizer_G, optimizer_D,
    style_lambda, identity_lambda,
    device, save_interval=1000
):

    Tensor = torch.FloatTensor

    const_noise = torch.empty((1, noise_channels)).uniform_(-1, 1)
    const_noise = const_noise.type(Tensor).to(device)
    const_noise = const_noise.repeat(eval_size, 1)
    const_theta = gen_theta(eval_size, random=False)
    const_theta = const_theta.type(Tensor).to(device)

    status = Status(max_iters)
    loss   = GANLoss()

    while status.batches_done < max_iters:

        for index, img in enumerate(dataset):

            # real image
            img = img.to(device)
            img = augment(img)
            # random variables
            z = sample_unoise((img.size(0), noise_channels), device)
            theta = gen_theta(batch_size)
            theta = theta.type(Tensor).to(device)

            '''
            Discriminator
            '''
            # D(x)
            real_prob,     _, real_logits = D(img)
            # D(G(z))
            fake = G(z, theta)
            fake = augment(fake)
            fake_prob, z_rec, fake_logits = D(fake.detach())

            adv_loss = loss.d_loss(real_prob, fake_prob)
            # style loss
            style_loss = style_criterion(fake_logits, real_logits, style_lambda)
            # identity loss
            id_loss = identity_criterion(z, z_rec, identity_lambda)
            # total
            D_loss = adv_loss + style_loss + id_loss

            # optimization
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()


            '''
            Generator
            '''
            # D(G(z))
            fake_prob, z_rec, fake_logits = D(fake)
            adv_loss = loss.g_loss(fake_prob)
            # identity loss
            id_loss = identity_criterion(z, z_rec, identity_lambda)
            # total
            G_loss = adv_loss + id_loss

            # optimization
            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

            # save sample images
            if status.batches_done % save_interval == 0:
                test_img = G(const_noise, const_theta)
                save_image(
                    test_img, f'./implementations/HoloGAN/result/{status.batches_done}.png',
                    nrow=10, normalize=True, value_range=(-1, 1))
                torch.save(
                    G.state_dict(),
                    f'implementations/HoloGAN/result/G_{status.batches_done}.pt')

            # updates
            loss_dict = dict(
                G=G_loss.item() if not torch.isnan(G_loss).any() else 0,
                D=D_loss.item() if not torch.isnan(D_loss).any() else 0
            )
            status.update(**loss_dict)

            if status.batches_done == max_iters:
                break

    status.plot_loss()


def style_criterion(fake_logits, real_logits, style_lambda=1.):
    criterion = nn.BCEWithLogitsLoss()
    losses = [style_lambda *                                         \
             (criterion(f, torch.zeros(f.size(), device=f.device)) + \
              criterion(r, torch.ones( r.size(), device=r.device))) for f, r in zip(fake_logits, real_logits)]
    return sum(losses)

def identity_criterion(z, z_reconstruct, identity_lambda=1):
    return identity_lambda * ((z_reconstruct - z) ** 2).mean()

def main(parser):

    # parser = add_arguments(parser)
    parser = add_args(parser,
        dict(
            g_channels      = [512, 'base channel width'],
            d_channels      = [64, 'base channel width'],
            latent_dim      = [128, 'input latent dimension'],
            activation      = ['lrelu', 'activation function name'],
            lr              = [0.0001, 'learning rate'],
            betas           = [[0.5, 0.999], 'betas'],
            policy          = ['color,translation', 'policy for diffaugment'],
            style_lambda    = [1., 'lambda for style loss'],
            identity_lambda = [1., 'lambda for identity loss'],
            eval_size       = [10, 'number of samples for eval']))
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)

    dataset = AnimeFace.asloader(
        args.batch_size, (args.image_size, args.min_year),
        pin_memory=not args.disable_gpu)

    G = Generator(args.g_channels, args.latent_dim, args.activation)
    D = Discriminator(args.d_channels, args.latent_dim, args.activation, args.image_size)

    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    augment = partial(DiffAugment, policy=args.policy)

    train(
        args.max_iters, args.batch_size, args.latent_dim, args.eval_size,
        dataset, augment,
        G, D, optimizer_G, optimizer_D,
        args.style_lambda, args.identity_lambda,
        device, 1000
    )
