
import os
import warnings

import torch
import torchvision as tv

from PIL import Image

# device specification
def get_device(): return torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# noise sampler
# normal distribution
def sample_nnoise(size, mean=0., std=1., device=None):
    if device == None:
        device = get_device()
    return torch.empty(size, device=device).normal_(mean, std)

# uniform distribution
def sample_unoise(size, from_=0., to=1., device=None):
    if device == None:
        device = get_device()
    return torch.empty(size, device=device).uniform_(from_, to)


class GANTrainingStatus:
    '''GAN training status helper'''
    def __init__(self):
        self.losses = {'G' : [], 'D' : []}
        self.batches_done = 0

    def append_g_loss(self, g_loss):
        '''append generator loss'''
        if isinstance(g_loss, torch.Tensor):
            g_loss = g_loss.item()
        self.losses['G'].append(g_loss)
    def append_d_loss(self, d_loss):
        '''append discriminator loss'''
        if isinstance(d_loss, torch.Tensor):
            d_loss = d_loss.item()
        self.losses['D'].append(d_loss)
        self.batches_done += 1

    def add_loss(self, key):
        '''define additional loss'''
        if not isinstance(key, str):
            raise Exception('Input a String object as the key. Got type {}'.format(type(key)))
        self.losses[key] = []
    def append_additional_loss(self, **kwargs):
        '''append additional loss'''
        for key, value in kwargs.items():
            try:
                self.losses[key].append(value)
            except KeyError as ke:
                warnings.warn('You have tried to append a loss keyed as \'{}\' that is not defined. Please call add_loss() or check the spelling.'.format(key))

    def append(self, g_loss, d_loss, **kwargs):
        '''append loss at once'''
        self.append_g_loss(g_loss)
        self.append_d_loss(d_loss)
        self.append_additional_loss(**kwargs)

    def plot_loss(self, filename='./loss.png'):
        '''plot loss'''
        import matplotlib
        matplotlib.use('agg')
        import matplotlib.pyplot as plt

        add_loss = [key for key in self.losses if key not in ['G', 'D']]
        G_loss = self.losses['G']
        D_loss = self.losses['D']

        plt.figure(figsize=(12, 8))
        for key in add_loss:
            plt.plot(self.losses[key])
        plt.plot(G_loss)
        plt.plot(D_loss)
        plt.title('Model Loss')
        plt.xlabel('iter')
        plt.ylabel('loss')
        plt.legend([key for key in add_loss] + ['Generator', 'Discriminator'], loc='upper left')

        plt.tight_layout()
        plt.savefig(filename)
        plt.close()

    def save_image(self, folder, G, *input, ema=False, filename=None, nrow=5, normalize=True, range=(-1, 1)):
        '''simple save image
        save_image func with
        sampling images, and only args that I use frequently
        '''
        G.eval()
        with torch.no_grad():
            images = G(*input)
        if not ema:
            G.train()

        if filename == None:
            filename = '{}.png'.format(self.batches_done)

        tv.utils.save_image(
            images, os.path.join(folder, filename), nrow=nrow, normalize=normalize, range=range
        )

    def __str__(self):
        '''print the latest losses when calling print() on the object'''
        partial_msg = []
        partial_msg += [
            '{:6}'.format(self.batches_done),
            '[D Loss : {:.5f}]'.format(self.losses['D'][-1]),
            '[G Loss : {:.5f}]'.format(self.losses['G'][-1]),
        ]
        # verbose additinal loss
        add_loss = [key for key in self.losses if key not in ['D', 'G']]
        if len(add_loss) > 0:
            for key in add_loss:
                if self.losses[key] == []: # skip when no entry
                    continue
                partial_msg.append(
                    '[{} : {:.5f}]'.format(key, self.losses[key][-1])
                )
        return '\t'.join(partial_msg)
        
def gif_from_files(image_paths, filename='out.gif', optimize=False, duration=500, loop=0):
    images = []
    for path in image_paths:
        images.append(Image.open(str(path)))
    images[0].save(filename, save_all=True, append_images=images[1:], optimize=optimize, duration=duration, loop=loop)

if __name__ == "__main__":
    '''TEST'''
    # device = get_device()
    # print(sample_nnoise((3, 64), 0, 0.02).size())
    # print(sample_nnoise((3, 64), 0, 0.02, torch.device('cpu')).size())
    # print(sample_nnoise((3, 64), 0, 0.02, device).size())
    # print(sample_unoise((3, 64), 0, 3).size())
    # print(sample_unoise((3, 64), 0, 3, torch.device('cpu')).size())
    # print(sample_unoise((3, 64), 0, 3, device).size())

    status = GANTrainingStatus()

    status.add_loss('real')
    status.add_loss('fake')

    import math

    for i in range(100):
        status.append(math.sin(i), math.cos(i), real=-i/10, fake=i/10)
        print(status)

    status.plot_loss()