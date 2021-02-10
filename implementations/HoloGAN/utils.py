
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

from torchvision.utils import save_image

from .model import Generator, Discriminator
from ..general import AnimeFaceDataset, to_loader
from ..gan_utils import DiffAugment

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
    epochs, batch_size, noise_channels, eval_size,
    dataset, DiffAugment, policy,
    G, D, optimizer_G, optimizer_D,
    gan_criterion, style_lambda, identity_lambda,
    device,
    save_interval=1000, verbose_interval=1000
):

    Tensor = torch.FloatTensor

    const_noise = torch.empty((1, noise_channels)).uniform_(-1, 1)
    const_noise = const_noise.type(Tensor).to(device)
    const_noise = const_noise.repeat(eval_size, 1)
    const_theta = gen_theta(eval_size, random=False)
    const_theta = const_theta.type(Tensor).to(device)

    batches_done = 0

    if DiffAugment:
        aug = DiffAugment
    else:
        aug = lambda x : x

    for epoch in range(epochs):

        for index, img in enumerate(dataset):

            # real image
            img = img.type(Tensor).to(device)
            img = aug(img, policy)
            # random variables
            z = torch.empty((batch_size, noise_channels)).uniform_(-1, 1)
            z = z.type(Tensor).to(device)
            theta = gen_theta(batch_size)
            theta = theta.type(Tensor).to(device)

            '''
            Discriminator
            '''
            # D(x)
            real_prob,     _, real_logits = D(img)
            real_loss = gan_criterion(real_prob, torch.ones(real_prob.size(), device=device))
            # D(G(z))
            fake = G(z, theta)
            fake = aug(fake, policy)
            fake_prob, z_rec, fake_logits = D(fake.detach())
            fake_loss = gan_criterion(fake_prob, torch.zeros(fake_prob.size(), device=device))
            # style loss
            style_loss = style_criterion(fake_logits, real_logits, style_lambda)
            # identity loss
            id_loss = identity_criterion(z, z_rec, identity_lambda)
            # total
            D_loss = real_loss + fake_loss + style_loss + id_loss

            # optimization
            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()


            '''
            Generator
            '''
            # D(G(z))
            fake = G(z, theta)
            fake = aug(fake, policy)
            fake_prob, z_rec, fake_logits = D(fake)
            fake_loss = gan_criterion(fake_prob, torch.ones(fake_prob.size(), device=device))
            # identity loss
            id_loss = identity_criterion(z, z_rec, identity_lambda)
            # total
            G_loss = fake_loss + id_loss

            # optimization
            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()


            batches_done += 1
            # verbose
            if batches_done % verbose_interval == 0 or batches_done == 1:
                print('{:6} [{:4} / {:4}] [G LOSS : {:.5f}] [D LOSS : {:.5f}]'.format(batches_done, epoch, epochs, G_loss.item(), D_loss.item()))
            # save sample images
            if batches_done % save_interval == 0 or batches_done == 1:
                test_img = G(const_noise, const_theta)
                save_image(test_img, './implementations/HoloGAN/result/{}.png'.format(batches_done), nrow=10, normalize=True, range=(-1, 1))


def style_criterion(fake_logits, real_logits, style_lambda=1.):
    criterion = nn.BCEWithLogitsLoss()
    losses = [style_lambda *                                         \
             (criterion(f, torch.zeros(f.size(), device=f.device)) + \
              criterion(r, torch.ones( r.size(), device=r.device))) for f, r in zip(fake_logits, real_logits)]
    return sum(losses)

def identity_criterion(z, z_reconstruct, identity_lambda=1):
    return identity_lambda * ((z_reconstruct - z) ** 2).mean()

def main(parser):
    # parameters
    # data
    image_size = 128
    batch_size = 32

    # model
    g_channels = 512
    d_channels = 64
    noise_channels = 128
    activation = 'lrelu'
    # activation = 'relu'

    # training
    epochs = 150
    lr = 0.0001
    betas = (0.5, 0.999)
    policy = 'color,translation'
    style_lambda = 1.
    identity_lambda = 1.

    # eval
    eval_size = 10


    dataset = AnimeFaceDataset(image_size)
    dataset = to_loader(dataset, batch_size)

    G = Generator(g_channels, noise_channels, activation)
    D = Discriminator(d_channels, noise_channels, activation, image_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    G.to(device)
    D.to(device)

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=betas)

    gan_criterion = nn.BCEWithLogitsLoss()
    
    train(
        epochs, batch_size, noise_channels, eval_size,
        dataset, DiffAugment, policy,
        G, D, optimizer_G, optimizer_D,
        gan_criterion, style_lambda, identity_lambda,
        device,
        1000, 1000
    )
