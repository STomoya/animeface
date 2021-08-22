
from functools import partial

import torch
import torch.optim as optim
import torchvision.transforms as transforms
from torchvision.utils import save_image

from dataset import AnimeFace, to_loader
from utils import Status, save_args, add_args
from nnutils.loss import WGANLoss, gradient_penalty
from nnutils import get_device, sample_nnoise

from .config import *
from .model import Generator, Discriminator

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

def train(
    latent_dim, dataset, to_loader,
    G, D,
    gp_lambda, drift_epsilon,
    device, save_interval=1000
):
    # NOTE: n_critic = 1

    training_status = Step()
    optimizer_G = get_optimizer(G, resl2lr[training_status.current_resolution])
    optimizer_D = get_optimizer(D, resl2lr[training_status.current_resolution])

    max_iters = 0
    for v in resl2num:
        max_iters += len(dataset) * v

    status = Status(max_iters)
    loss   = WGANLoss()
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

            real_prob = D(real, D_mode)
            fake_prob = D(fake.detach(), D_mode)
            adv_loss = loss.d_loss(real_prob, fake_prob)
            gp_loss = gp(real, fake, partial(D, phase=D_mode), None)
            drift = real_prob.square().mean()
            D_loss = adv_loss + (gp_lambda * gp_loss) + (drift_epsilon * drift)

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


            G.update_alpha(delta, G_mode)
            D.update_alpha(delta, D_mode)

            if status.batches_done % save_interval == 0:
                save_image(
                    fake, f'implementations/PGGAN/result/{status.batches_done}.png',
                    nrow=6 , normalize=True, value_range=(-1, 1))

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

def add_arguments(parser):
    parser.add_argument('--loss-type', default='wgan_gp', choices=['wgan_gp', 'lsgan'], help='loss type')
    parser.add_argument('--latent-dim', default=100, type=int, help='dimension of input latent')
    parser.add_argument('--gp-lambda', default=10, type=float, help='lambda for gradient penalty')
    parser.add_argument('--drift-epsilon', default=0.001, type=float, help='epsilon for drift')
    return parser

def main(parser):

    # parser = add_arguments(parser)
    parser = add_args(parser,
        dict(
            latent_dim=[100, 'input latent dimension'],
            gp_lambda=[10., 'lambda for gradient penalty'],
            drift_epsilon=[0.001, 'eps for drift']))
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

    G = Generator(latent_dim=args.latent_dim)
    D = Discriminator()

    G.to(device)
    D.to(device)

    train(
        args.latent_dim, dataset, to_loader,
        G, D, args.gp_lambda, args.drift_epsilon,
        device, args.save
    )
