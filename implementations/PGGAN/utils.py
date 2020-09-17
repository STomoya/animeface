
import torch
import torch.nn as nn
import torch.optim as optim
import torch.autograd as autograd
import torchvision.transforms as transforms
from torchvision.utils import save_image
import numpy as np

from .config import *
from .model import Generator, Discriminator

from ..general import AnimeFaceDataset, to_loader

class Step:
    '''
    manage training phases

    initual : D stablization
    otherwise : G transition -> G stablization -> D transition -> D stablization
    '''
    def __init__(self):
        self.phases = {
            'G_transition' : {
                'G' : 't',
                'D' : 's'
            },
            'D_transition' : {
                'G' : 's',
                'D' : 't'
            },
            'G_stablization' : {
                'G' : 's',
                'D' : 's'
            },
            'D_stablization' : {
                'G' : 's',
                'D' : 's'
            },
        }
        self.current_phase = 'D_stablization'

        self.current_resolution = 4
        self.max_resolution = 128

        self.skip_count = 1

        self.grow_flag = False
    
    def step(self):
        if not self.skip_count >= resl2num[self.current_resolution]:
            self.skip_count += 1
            return True
        self.skip_count = 1

        if self.current_phase == 'D_stablization':
            self.current_phase = 'G_transition'
            self.current_resolution *= 2
            self.grow_flag = True
        elif self.current_phase == 'G_transition':
            self.current_phase = 'G_stablization'
        elif self.current_phase == 'G_stablization':
            self.current_phase = 'D_transition'
        elif self.current_phase == 'D_transition':
            self.current_phase = 'D_stablization'
        
        if self.current_resolution > self.max_resolution:
            return False
        else:
            return True

    def get_mode(self):
        g_mode = self.phases[self.current_phase]['G']
        d_mode = self.phases[self.current_phase]['D']
        return g_mode, d_mode

    def should_grow(self):
        if self.grow_flag:
            self.grow_flag = False
            return True
        else:
            False

def calc_alpha_delta(length, resl):
    return 1. / (length // resl2batch_size[resl]) * resl2num[resl]

def get_optimizer(net, lr):
    return optim.Adam(net.parameters(), lr=lr, betas=betas)

def calc_magnitude(last_d, real_prob):
    d = 0.1 * real_prob.mean().item() + 0.9 * last_d
    magnitude = 0.2 * (max(0, d - 0.5) ** 2)
    return magnitude, d

def train_lsgan(
    latent_dim,
    G,
    D,
    criterion,
    dataset,
    to_loader,
    device,
    verbose_interval=1000,
    save_interval=1000
):

    Tensor = torch.FloatTensor

    last_d = 0

    losses = {
        'D' : [],
        'G' : []
    }
    training_status = Step()
    optimizer_G = get_optimizer(G, resl2lr[training_status.current_resolution])
    optimizer_D = get_optimizer(D, resl2lr[training_status.current_resolution])
    
    batches_done = 0

    while True:
        G_mode, D_mode = training_status.get_mode()

        dataset.update_transform(training_status.current_resolution)
        dataloader = to_loader(dataset, resl2batch_size[training_status.current_resolution])

        delta = calc_alpha_delta(len(dataset), training_status.current_resolution)

        for index, image in enumerate(dataloader):
            # label smoothing
            valid = (1.0 - 0.7) * np.random.randn(image.size(0), 1) + 0.7
            valid = torch.from_numpy(valid).type(Tensor).to(device)
            fake  = (0.3 - 0.0) * np.random.randn(image.size(0), 1) + 0.0
            fake  = torch.from_numpy(fake).type(Tensor).to(device)

            z = np.random.normal(0, 1, (image.size(0), latent_dim))
            z = torch.from_numpy(z).type(Tensor).to(device)

            image = image.type(Tensor).to(device)

            # generate image
            fake_image = G(z, G_mode)

            '''
            Train Discriminator
            '''

            real_prob = D(image, D_mode)
            fake_prob = D(fake_image.detach(), D_mode)
            real_loss = criterion(real_prob, valid)
            fake_loss = criterion(fake_prob, fake)
            d_loss = (real_loss + fake_loss) / 2

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            magnitude, last_d = calc_magnitude(last_d, real_prob=real_prob)
            D.update_noise(magnitude)

            '''
            Train Generator
            '''

            fake_prob = D(fake_image, D_mode)
            g_loss = criterion(fake_prob, valid)

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()


            G.update_alpha(delta, G_mode)
            D.update_alpha(delta, D_mode)

            losses['G'].append(g_loss.item())
            losses['D'].append(d_loss.item())



            if batches_done % verbose_interval == 0:
                print('{:10} {:14} G LOSS : {:.5f}, D LOSS {:.5f}'.format(batches_done, training_status.current_phase, losses['G'][-1], losses['D'][-1]))

            if batches_done % save_interval == 0:
                save_image(fake_image, './implementations/PGGAN/result/{}.png'.format(batches_done), nrow=6 , normalize=True)

            batches_done += 1


        is_training = training_status.step()
        if not is_training:
            break

        if training_status.should_grow():
            G.grow()
            D.grow()
            optimizer_G = get_optimizer(G, resl2lr[training_status.current_resolution])
            optimizer_D = get_optimizer(D, resl2lr[training_status.current_resolution])
            G.to(device)
            D.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def train_wgangp(
    latent_dim,
    G,
    D,
    gp_lambda,
    drift_epsilon,
    dataset,
    to_loader,
    device,
    verbose_interval=1000,
    save_interval=1000
):
    # NOTE: n_critic = 1

    Tensor = torch.FloatTensor

    losses = {
        'D' : [],
        'G' : []
    }
    training_status = Step()
    optimizer_G = get_optimizer(G, resl2lr[training_status.current_resolution])
    optimizer_D = get_optimizer(D, resl2lr[training_status.current_resolution])
    
    batches_done = 0

    while True:
        G_mode, D_mode = training_status.get_mode()

        dataset.update_transform(training_status.current_resolution)
        dataloader = to_loader(dataset, resl2batch_size[training_status.current_resolution])

        delta = calc_alpha_delta(len(dataset), training_status.current_resolution)

        for index, image in enumerate(dataloader):

            z = np.random.normal(0, 1, (image.size(0), latent_dim))
            z = torch.from_numpy(z).type(Tensor).to(device)

            image = image.type(Tensor).to(device)

            # generate image
            fake_image = G(z, G_mode)

            '''
            Train Discriminator
            '''

            real_prob = D(image, D_mode)
            fake_prob = D(fake_image.detach(), D_mode)
            gp = calc_gradient_penalty(D, image, fake_image, D_mode, device)
            drift = real_prob.square().mean()
            d_loss = - real_prob.mean() + fake_prob.mean() + (gp_lambda * gp) + (drift_epsilon * drift)

            optimizer_D.zero_grad()
            d_loss.backward(retain_graph=True)
            optimizer_D.step()

            '''
            Train Generator
            '''

            fake_prob = D(fake_image, D_mode)
            g_loss = - fake_prob.mean()

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()


            G.update_alpha(delta, G_mode)
            D.update_alpha(delta, D_mode)

            losses['G'].append(g_loss.item())
            losses['D'].append(d_loss.item())



            if batches_done % verbose_interval == 0:
                print('{:10} {:14} G LOSS : {:.5f}, D LOSS {:.5f}'.format(batches_done, training_status.current_phase, losses['G'][-1], losses['D'][-1]))

            if batches_done % save_interval == 0:
                save_image(fake_image, './implementations/PGGAN/result/{}.png'.format(batches_done), nrow=6 , normalize=True)

            batches_done += 1


        is_training = training_status.step()
        if not is_training:
            break

        if training_status.should_grow():
            G.grow()
            D.grow()
            optimizer_G = get_optimizer(G, resl2lr[training_status.current_resolution])
            optimizer_D = get_optimizer(D, resl2lr[training_status.current_resolution])
            G.to(device)
            D.to(device)
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

def calc_gradient_penalty(D, real_image, fake_image, mode, device):
    alpha = torch.from_numpy(np.random.random((real_image.size(0), 1, 1, 1)))
    alpha = alpha.type(torch.FloatTensor).to(device)

    interpolates = (alpha * real_image + ((1 - alpha) * fake_image)).requires_grad_(True)
    d_interpolates = D(interpolates, mode)

    fake = torch.Tensor(real_image.size(0), 1).fill_(1.).requires_grad_(False)
    fake = fake.type(torch.FloatTensor).to(device)

    gradients = autograd.grad(
        outputs=d_interpolates.view(d_interpolates.size(0), 1),
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    penalty = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return penalty

def main():

    loss_type = 'lsgan'
    loss_type = 'wgan_gp'

    latent_dim = 100
    # params for wgan_gp
    gp_lambda = 10
    drift_epsilon = 0.001

    # add function for updating transform
    class AnimeFaceDatasetAlpha(AnimeFaceDataset):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)
        def update_transform(self, resolution):
            self.transform = transforms.Compose([
                transforms.Resize((resolution, resolution)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
            ])
    
    dataset = AnimeFaceDatasetAlpha(image_size=4)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    G = Generator(latent_dim=latent_dim)
    D = Discriminator(loss_type=loss_type)

    G.to(device)
    D.to(device)

    if loss_type == 'lsgan':
        criterion = nn.MSELoss()

        train_lsgan(
            latent_dim=latent_dim,
            G=G,
            D=D,
            criterion=criterion,
            dataset=dataset,
            to_loader=to_loader,
            device=device,
        )
    else:
        train_wgangp(
            latent_dim=latent_dim,
            G=G,
            D=D,
            gp_lambda=gp_lambda,
            drift_epsilon=drift_epsilon,
            dataset=dataset,
            to_loader=to_loader,
            device=device
        )
