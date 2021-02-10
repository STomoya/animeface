
'''
TODO: train the model
        This model takes too much memory and time to train.
        I will train the model when I have time or more computational power...
'''

import itertools

import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from .model import Generator, Discriminator

from ..general import GeneratePairImageDanbooruDataset, to_loader

def train(
    epochs,
    dataset,
    G_A2B, G_B2A,
    local_DA, local_DB,
    global_DA, global_DB,
    optimizer_G,
    optimizer_D,
    L1Loss,
    MSELoss,
    BCELoss,
    device,
    verbose_interval=1000,
    save_interval=1000
):

    Tensor = torch.FloatTensor
    global_patch = (1,  6,  6)
    local_patch  = (1, 30, 30)

    batches_done = 0

    for epoch in range(epochs):
        for index, (A, B) in enumerate(dataset):
            A = A.type(Tensor).to(device)
            B  = B.type(Tensor).to(device)

            # label smoothing
            local_real = ((1.0 - 0.7) * torch.rand(A.size(0), *local_patch) + 0.7).to(device)
            local_fake = ((0.3 - 0.0) * torch.rand(A.size(0), *local_patch) + 0.0).to(device)
            global_real = ((1.0 - 0.7) * torch.rand(A.size(0), *global_patch) + 0.7).to(device)
            global_fake = ((0.3 - 0.0) * torch.rand(A.size(0), *global_patch) + 0.0).to(device)

            '''
            Discriminator
            '''

            # gen image
            A2B, _ = G_A2B(A)
            B2A, _ = G_B2A(B)

            # discriminate
            # real
            A_local,  A_local_cam  = local_DA(A)
            A_global, A_global_cam = global_DA(A)
            B_local,  B_local_cam  = local_DB(B)
            B_global, B_global_cam = global_DB(B)
            # fake
            B2A_local,  B2A_local_cam  = local_DA(B2A)
            B2A_global, B2A_global_cam = global_DA(B2A)
            A2B_local,  A2B_local_cam  = local_DB(A2B)
            A2B_global, A2B_global_cam = global_DB(A2B)

            # cam label
            cam_local_real = ((1.0 - 0.7) * torch.rand(*A_local_cam.size()) + 0.7).to(device)
            cam_local_fake = ((0.3 - 0.0) * torch.rand(*A_local_cam.size()) + 0.0).to(device)
            cam_global_real = ((1.0 - 0.7) * torch.rand(*A_global_cam.size()) + 0.7).to(device)
            cam_global_fake = ((0.3 - 0.0) * torch.rand(*A_global_cam.size()) + 0.0).to(device)

            # losses
            A_local_loss      = MSELoss(A_local,      local_real)      + MSELoss(B2A_local,      local_fake)
            A_local_cam_loss  = MSELoss(A_local_cam,  cam_local_real)  + MSELoss(B2A_local_cam,  cam_local_fake)
            A_global_loss     = MSELoss(A_global,     global_real)     + MSELoss(B2A_global,     global_fake)
            A_global_cam_loss = MSELoss(A_global_cam, cam_global_real) + MSELoss(B2A_global_cam, cam_global_fake)
            B_local_loss      = MSELoss(B_local,      local_real)      + MSELoss(A2B_local,      local_fake)
            B_local_cam_loss  = MSELoss(B_local_cam,  cam_local_real)  + MSELoss(A2B_local_cam,  cam_local_fake)
            B_global_loss     = MSELoss(B_global,     global_real)     + MSELoss(A2B_global,     global_fake)
            B_global_cam_loss = MSELoss(B_global_cam, cam_global_real) + MSELoss(A2B_global_cam, cam_global_fake)

            A_loss = A_local_loss + A_local_cam_loss + A_global_loss + A_global_cam_loss
            B_loss = B_local_loss + B_local_cam_loss + B_global_loss + B_global_cam_loss
            D_loss = A_loss + B_loss

            optimizer_D.zero_grad()
            D_loss.backward()
            optimizer_D.step()

            '''
            Generator
            '''

            # gen image
            A2B, A2B_cam = G_A2B(A)
            B2A, B2A_cam = G_B2A(B)

            # re-gen image
            A2B2A, _ = G_B2A(A2B)
            B2A2B, _ = G_A2B(B2A)

            # identity
            A2A, A2A_cam = G_B2A(A)
            B2B, B2B_cam = G_A2B(B)

            # fool discriminator
            B2A_local,  B2A_local_cam  = local_DA(B2A)
            B2A_global, B2A_global_cam = global_DA(B2A)
            A2B_local,  A2B_local_cam  = local_DB(A2B)
            A2B_global, A2B_global_cam = global_DB(A2B)

            # generator cam label
            cam_real = ((1.0 - 0.7) * torch.rand(*A2B_cam.size()) + 0.7).to(device)
            cam_fake = ((0.3 - 0.0) * torch.rand(*A2B_cam.size()) + 0.0).to(device)

            # losses
            # adversarial
            A_local_loss      = MSELoss(B2A_local,      local_real)
            A_local_cam_loss  = MSELoss(B2A_local_cam,  cam_local_real)
            A_global_loss     = MSELoss(B2A_global,     global_real)
            A_global_cam_loss = MSELoss(B2A_global_cam, cam_global_real)
            B_local_loss      = MSELoss(A2B_local,      local_real)
            B_local_cam_loss  = MSELoss(A2B_local_cam,  cam_local_real)
            B_global_loss     = MSELoss(A2B_global,     global_real)
            B_global_cam_loss = MSELoss(A2B_global_cam, cam_global_real)

            # cycle
            A2B2A_loss = L1Loss(A2B2A, A)
            B2A2B_loss = L1Loss(B2A2B, B)

            # identity
            A2A_loss = L1Loss(A2A, A)
            B2B_loss = L1Loss(B2B, B)

            A2A_cam_loss = BCELoss(B2A_cam, cam_real) + BCELoss(A2A_cam, cam_fake)
            B2B_cam_loss = BCELoss(A2B_cam, cam_real) + BCELoss(B2B_cam, cam_fake)

            A_loss = A_local_loss + A_local_cam_loss + A_global_loss + A_global_cam_loss + 10 * A2B2A_loss + 10 * A2A_loss + 1000 * A2A_cam_loss
            B_loss = B_local_loss + B_local_cam_loss + B_global_loss + B_global_cam_loss + 10 * B2A2B_loss + 10 * B2B_loss + 1000 * B2B_cam_loss
            G_loss = A_loss + B_loss

            optimizer_G.zero_grad()
            G_loss.backward()
            optimizer_G.step()


            if batches_done % verbose_interval == 0:
                print('{:6} G Loss : {:.5f} D Loss : {:.5f}'.format(batches_done, G_loss.item(), D_loss.item()))

            if batches_done % save_interval == 0:
                image_grid = make_grid(A, A2B, B, n_image=999)
                save_image(image_grid, './implementations/UGATIT/result/{}.png'.format(batches_done), nrow=9, normalize=True)
                
            batches_done += 1


def make_grid(A, A2B, B, n_image=999):
    b = A.size(0)
    split_A = A.chunk(b)
    split_A2B = A2B.chunk(b)
    split_B = B.chunk(b)

    images = []
    for index, (a, a2b, b) in enumerate(zip(split_A, split_A2B, split_B)):
        images += [a, a2b, b]
        if index == n_image:
            break

    image_grid = torch.cat(images, dim=0)
    return image_grid

from PIL import ImageFilter

class SoftMozaic(object):
    def __init__(self, linear_scale, radius):
        self.linear_scale = linear_scale
        self.radius = radius

    def __call__(self, sample):
        sample = sample.filter(ImageFilter.GaussianBlur(self.radius))
        return sample.resize([int(x / self.linear_scale) for x in sample.size]).resize(sample.size)

def main(parser):
    import torchvision.transforms as T

    epochs = 5

    lr = 0.0001
    betas = (0.5, 0.999)
    batch_size = 5

    pair_trasform = SoftMozaic(1.3, 0.4)

    dataset = GeneratePairImageDanbooruDataset(pair_trasform, image_size=128)
    dataset.co_transform_head = T.Compose([
        T.CenterCrop((350, 350)),
        T.Resize(128)
    ])
    dataset = to_loader(dataset, batch_size)

    device = torch.device('cpu')
    # device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    G_A2B = Generator(n_blocks=6)
    G_B2A = Generator(n_blocks=6)
    local_DA = Discriminator(n_layers=3)
    local_DB = Discriminator(n_layers=3)
    global_DA = Discriminator(n_layers=5)
    global_DB = Discriminator(n_layers=5)

    G_A2B.to(device)
    G_B2A.to(device)
    local_DA.to(device)
    local_DB.to(device)
    global_DA.to(device)
    global_DB.to(device)

    optimizer_G = optim.Adam(itertools.chain(G_A2B.parameters(), G_B2A.parameters()), lr=lr, betas=betas)
    optimizer_D = optim.Adam(itertools.chain(local_DA.parameters(), local_DB.parameters(), global_DA.parameters(), global_DB.parameters()), lr=lr, betas=betas)
    
    MSELoss = nn.MSELoss()
    L1Loss = nn.L1Loss()
    BCELoss = nn.BCEWithLogitsLoss()

    torch.cuda.empty_cache()

    train(
        epochs, dataset, G_A2B, G_B2A,
        local_DA, local_DB, global_DA, global_DB,
        optimizer_G, optimizer_D,
        L1Loss, MSELoss, BCELoss,
        device, 1000, 1000
    )