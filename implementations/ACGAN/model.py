
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)


class Generator(nn.Module):
    def __init__(self, latent_dim, label_dim):
        super(Generator, self).__init__()

        self.label_embed = nn.Embedding(label_dim, latent_dim)

        self.conv = nn.Sequential(
            nn.ConvTranspose2d(latent_dim, 1024, 4, 1, 0, bias=False),
            nn.BatchNorm2d(1024),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(1024, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(512, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(256, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(128, 64, 4, 2, 1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.ConvTranspose2d(64, 3, 4, 2, 1, bias=False),
            nn.Tanh()
        )
    
    def forward(self, noise, label):
        label = self.label_embed(label)
        x = torch.mul(noise, label)
        x = x.view(x.size(0), x.size(1), 1, 1)
        return self.conv(x)



class Discriminator(nn.Module):
    def __init__(self, label_dim):
        super(Discriminator, self).__init__()
        self.label_dim = label_dim

        self.conv = nn.Sequential(
            nn.Conv2d(3, 64, 4, 2, 1, bias=False),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(64, 128, 4, 2, 1, bias=False),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(128, 256, 4, 2, 1, bias=False),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(256, 512, 4, 2, 1, bias=False),
            nn.BatchNorm2d(512),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(512, 1024, 4, 2, 1, bias=False),
            nn.BatchNorm2d(1024),
            nn.LeakyReLU(0.2, inplace=True)
        )

        self.validity_layer = nn.Conv2d(1024, 1, 4, 1, 0, bias=False)
        
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.label_layer = nn.Linear(1024, label_dim)

    def forward(self, img):
        x = self.conv(img)
        validity = self.validity_layer(x)
        x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        label = self.label_layer(x)
        return validity, label

if __name__ == "__main__":
    model = Generator(100, 28)

    noise = torch.randn(32, 100)
    # label = torch.randn(32, 28)
    label = torch.from_numpy(np.random.randint(0, 28, (32, )))

    output = model(noise, label)

    print(output.size())

    model = Discriminator(28)

    output1, output2 = model(output)

    print(output1.size(), output2.size())