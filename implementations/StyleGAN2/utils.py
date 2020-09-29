
import torch
import torch.nn as nn
from torch.autograd import grad, Variable
import torch.optim as optim
from torchvision.utils import save_image

from .model import Generator, Discriminator

from ..general import AnimeFaceDataset, to_loader
from ..gan_utils import DiffAugment

def toggle_grad(model, state):
    for param in model.parameters():
        param.requires_grad = state

def update_ema(model_ema, model, decay=0.999):
    toggle_grad(model, False)

    param_ema = dict(model_ema.named_parameters())
    param     = dict(model.named_parameters())

    for key in param_ema.keys():
        param_ema[key].data.mul_(decay).add_(param[key].data, alpha=(1 - decay))

    toggle_grad(model, True)

def train(
    epochs,
    latent_dim,
    G,
    G_ema,
    D,
    optimizer_G,
    optimizer_D,
    gp_lambda,
    drift_epsilon,
    dataset,
    num_images,
    diffaug,
    policy,
    device,
    verbose_interval=1000,
    save_interval=1000
):

    Tensor = torch.FloatTensor
    softplus = nn.Softplus()

    losses = {
        'D' : [],
        'G' : []
    }

    batches_done = 0

    constant_latent = torch.randn(2, num_images, latent_dim).type(Tensor).to(device)
    constant_latent = constant_latent.unbind(0)

    for epoch in range(epochs):
        for index, image in enumerate(dataset):

            z = torch.randn(2, image.size(0), latent_dim).type(Tensor).to(device)
            z = z.unbind(0)

            image = image.type(Tensor).to(device)

            # gen image
            gen_image_ = G(z)

            image     = diffaug(image, policy)
            gen_image = diffaug(gen_image_, policy)

            '''
            Train Discriminator
            '''

            real_prob = D(image)
            fake_prob = D(gen_image.detach())
            gp = calc_gradient_penalty(image.detach(), D)
            drift = real_prob.square().mean()
            d_loss = softplus(-real_prob).mean() + softplus(fake_prob).mean() + (gp_lambda * gp) + (drift_epsilon * drift)

            optimizer_D.zero_grad()
            d_loss.backward()
            optimizer_D.step()

            '''
            Train Generator
            '''

            fake_prob = D(gen_image)
            g_loss = softplus(-fake_prob).mean()

            optimizer_G.zero_grad()
            g_loss.backward()
            optimizer_G.step()

            # EMA
            update_ema(G_ema, G)

            losses['G'].append(g_loss.item())
            losses['D'].append(d_loss.item())


            if batches_done % verbose_interval == 0:
                print('{:8} G LOSS : {:.5f}, D LOSS : {:.5f}'.format(batches_done, losses['G'][-1], losses['D'][-1]))
            if batches_done % save_interval == 0:
                G_ema.eval()
                with torch.no_grad():
                    gen_image = G_ema(constant_latent, injection=7)
                save_image(gen_image, './implementations/StyleGAN2/result/{}.png'.format(batches_done), nrow=6, normalize=True, range=(-1, 1))


            batches_done += 1

def calc_gradient_penalty(real_image, D):
    loc_real_image = Variable(real_image, requires_grad=True)
    gradients = grad(
        outputs=D(loc_real_image)[:, 0].sum(),
        inputs=loc_real_image,
        create_graph=True, retain_graph=True
    )[0]
    gradients = gradients.view(gradients.size(0), -1)
    gradients = (gradients ** 2).sum(dim=1).mean()
    return gradients / 2

def main():

    epochs = 100
    lr = 0.001
    betas = (0, 0.99)

    batch_size = 10
    num_images = 30

    latent_dim = 512
    gp_lambda = 10
    drift_epsilon = 0.001

    policy = 'color,translation'

    dataset = AnimeFaceDataset(image_size=128)
    dataset = to_loader(dataset, batch_size)

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    G = Generator()
    G_ema = Generator()
    toggle_grad(G_ema, False)
    D = Discriminator()

    G.to(device)
    G_ema.to(device)
    D.to(device)

    update_ema(G_ema, G, decay=0.)

    optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
    optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=betas)

    train(
        epochs, latent_dim, G, G_ema, D, optimizer_G, optimizer_D, gp_lambda, drift_epsilon,
        dataset, num_images, DiffAugment, policy, device, 1000, 1000
    )