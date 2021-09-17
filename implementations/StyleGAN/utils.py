
from functools import partial

import torch
import torch.optim as optim
from torch.autograd import grad, Variable
import torchvision.transforms as transforms
from torchvision.utils import save_image

from .config import *
from .model import Generator, Discriminator

from dataset import AnimeFace, to_loader
from utils import Status, save_args, add_args
from nnutils import sample_nnoise, get_device
from nnutils.loss import NonSaturatingLoss, gradient_penalty

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
    save_interval=1000
):
    # NOTE: n_critic = 1
    training_status = Step()
    optimizer_G = get_optimizer(G, resl2lr[training_status.current_resolution])
    optimizer_D = get_optimizer(D, resl2lr[training_status.current_resolution])

    max_iters = 0
    for v in resl2num:
        max_iters += len(dataset) * v
    status = Status(max_iters)
    loss   = NonSaturatingLoss()
    gp     = gradient_penalty()

    while True:
        G_mode, D_mode = training_status.get_mode()

        dataset.update_transform(training_status.current_resolution)
        dataloader = to_loader(dataset, resl2batch_size[training_status.current_resolution])

        delta = calc_alpha_delta(len(dataset), training_status.current_resolution)

        for index, image in enumerate(dataloader):

            z = sample_nnoise((image.size(0), latent_dim), device)
            real = image.to(device)

            # generate image
            fake = G(z, G_mode)

            '''
            Train Discriminator
            '''

            # D(real)
            real_prob = D(real, D_mode)
            # D(G(z))
            fake_prob = D(fake.detach(), D_mode)

            gp_loss = gp(real, fake, partial(D, phase=D_mode), None)
            drift = real_prob.square().mean()
            adv_loss = loss.d_loss(real_prob, fake_prob)

            D_loss = adv_loss + gp_loss * gp_lambda + drift * drift_epsilon

            optimizer_D.zero_grad()
            D_loss.backward(retain_graph=True)
            optimizer_D.step()

            '''
            Train Generator
            '''

            fake_prob = D(fake, D_mode)
            G_loss = loss.g_loss(fake_prob)

            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()

            if training_status.update_G_alpha():
                G.update_alpha(delta, G_mode)
            if training_status.update_D_alpha():
                D.update_alpha(delta, D_mode)

            if status.batches_done % save_interval == 0:
                save_image(fake, f'./implementations/StyleGAN/result/{status.batches_done}.png', nrow=6 , normalize=True)
                torch.save(G.state_dict(), f'./implementations/StyleGAN/result/G_{status.batches_done}.pt')

            status.update(d=D_loss.item(), g=G_loss.item())

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

    status.plot_loss()

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

def main(parser):

    # parser = add_arguments(parser)
    parser = add_args(parser,
        dict(
            latent_dim=[512, 'latent input dimension'],
            gp_lambda=[10., 'lambda for gradient penalty'],
            drift_epsilon=[0.0001, 'epsilon for drift']))
    args = parser.parse_args()
    save_args(args)

    # add function for updating transform
    class AnimeFaceDatasetAlpha(AnimeFace):
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

    device = get_device(not args.disable_gpu)

    G = Generator(style_dim=args.latent_dim)
    D = Discriminator()

    G.to(device)
    D.to(device)

    train_wgangp(
        args.latent_dim, G, D,
        args.gp_lambda, args.drift_epsilon,
        dataset, to_loader,
        device, args.save
    )
