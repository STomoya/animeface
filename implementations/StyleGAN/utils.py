
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad, Variable
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
                'D' : 't'
            },
            'D_transition' : {
                'G' : 's',
                'D' : 't'
            },
            'G_stablization' : {
                'G' : 's',
                'D' : 't'
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
    
    def update_G_alpha(self):
        if self.current_phase == 'G_transition':
            return True
        return False
    
    def update_D_alpha(self):
        if self.current_phase == 'D_transition':
            return True
        return False

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
    return 1. / ((length // resl2batch_size[resl]) * resl2num[resl])

def get_optimizer(net, lr):
    return optim.Adam(net.parameters(), lr=lr, betas=betas)

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
    softplus = nn.Softplus()

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
            gp = calc_gradient_penalty(image.detach(), D, D_mode)
            drift = real_prob.square().mean()
            d_loss = softplus(-real_prob).mean() + softplus(fake_prob).mean() + (gp_lambda * gp) + (drift_epsilon * drift)

            optimizer_D.zero_grad()
            d_loss.backward(retain_graph=True)
            optimizer_D.step()

            '''
            Train Generator
            '''

            fake_prob = D(fake_image, D_mode)
            g_loss = softplus(-fake_prob).mean()

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()


            if training_status.update_G_alpha():
                G.update_alpha(delta, G_mode)
            if training_status.update_D_alpha():
                D.update_alpha(delta, D_mode)

            losses['G'].append(g_loss.item())
            losses['D'].append(d_loss.item())



            if batches_done % verbose_interval == 0:
                print('{:10} {:14} G LOSS : {:.5f}, D LOSS {:.5f}'.format(batches_done, training_status.current_phase, losses['G'][-1], losses['D'][-1]))

            if batches_done % save_interval == 0:
                save_image(fake_image, './implementations/StyleGAN/result/{}.png'.format(batches_done), nrow=6 , normalize=True)

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

def calc_gradient_penalty(real_image, D, phase):
    loc_real_image = Variable(real_image, requires_grad=True)
    gradients = grad(
        outputs=D(loc_real_image, phase)[:, 0].sum(),
        inputs=loc_real_image,
        create_graph=True, retain_graph=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradients = (gradients ** 2).sum(dim=1).mean()
    return gradients / 2

def main():
    latent_dim = 512
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

    G = Generator(style_dim=latent_dim)
    D = Discriminator()

    G.to(device)
    D.to(device)

    train_wgangp(
        latent_dim=latent_dim,
        G=G,
        D=D,
        gp_lambda=gp_lambda,
        drift_epsilon=drift_epsilon,
        dataset=dataset,
        to_loader=to_loader,
        device=device,
        # verbose_interval=1
    )
