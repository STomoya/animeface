
import torch
import torch.nn as nn
import torch.nn.functional as F

def weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find("Conv") != -1:
        torch.nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find("BatchNorm2d") != -1:
        torch.nn.init.normal_(m.weight.data, 1.0, 0.02)
        torch.nn.init.constant_(m.bias.data, 0.0)

class Down(nn.Module):
    def __init__(self, in_channel, out_channel, normalize=True, leaky=True, drop_out=0.0):
        super(Down, self).__init__()

        layers = [
            nn.Conv2d(in_channel, out_channel, 4, 2, 1)
        ]
        if normalize:
            layers.append(nn.InstanceNorm2d(out_channel))
        if leaky:
            layers.append(nn.LeakyReLU(0.2))
        else:
            layers.append(nn.ReLU(inplace=True))
        if drop_out:
            layers.append(nn.Dropout(drop_out))

        self.conv = nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv(x)
        return x

class Up(nn.Module):
    def __init__(self, in_channel, out_channel, drop_out=0.0):
        super(Up, self).__init__()
        layers = [
            nn.ConvTranspose2d(in_channel, out_channel, 4, 2, 1),
            nn.InstanceNorm2d(out_channel),
            nn.ReLU(inplace=True)
        ]
        if drop_out:
            layers.append(nn.Dropout(drop_out))

        self.conv = nn.Sequential(*layers)
    
    def forward(self, x):
        x = self.conv(x)
        return x

class Generator(nn.Module):
    def __init__(self, drop_out=0.0):
        super(Generator, self).__init__()

        self.d1 = Down(3, 64, normalize=False)
        self.d2 = Down(64, 128)
        self.d3 = Down(128, 256)
        self.d4 = Down(256, 512, drop_out=drop_out)
        self.d5 = Down(512, 512, drop_out=drop_out)
        self.d6 = Down(512, 512, drop_out=drop_out)
        self.d7 = Down(512, 512, drop_out=drop_out)
        self.d8 = Down(512, 512, normalize=False, leaky=False, drop_out=drop_out)

        self.u1 = Up(512, 512, drop_out=drop_out)
        self.u2 = Up(512*2, 512, drop_out=drop_out)
        self.u3 = Up(512*2, 512, drop_out=drop_out)
        self.u4 = Up(512*2, 512, drop_out=drop_out)
        self.u5 = Up(512*2, 256)
        self.u6 = Up(256*2, 128)
        self.u7 = Up(128*2, 64)
        self.u8 = nn.Sequential(
            nn.ConvTranspose2d(64*2, 3, 4, 2, 1),
            nn.Tanh()
        )
    
    def forward(self, x):

        d_out_1 = self.d1(x)
        d_out_2 = self.d2(d_out_1)
        d_out_3 = self.d3(d_out_2)
        d_out_4 = self.d4(d_out_3)
        d_out_5 = self.d5(d_out_4)
        d_out_6 = self.d6(d_out_5)
        d_out_7 = self.d7(d_out_6)
        d_out_8 = self.d8(d_out_7)

        x = self.u1(d_out_8)
        x = self.u2(torch.cat([x, d_out_7], dim=1))
        x = self.u3(torch.cat([x, d_out_6], dim=1))
        x = self.u4(torch.cat([x, d_out_5], dim=1))
        x = self.u5(torch.cat([x, d_out_4], dim=1))
        x = self.u6(torch.cat([x, d_out_3], dim=1))
        x = self.u7(torch.cat([x, d_out_2], dim=1))
        x = self.u8(torch.cat([x, d_out_1], dim=1))

        return x

class Discriminator(nn.Module):
    def __init__(self):
        super(Discriminator, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(3*2, 64, 4, 2, 1),
            nn.LeakyReLU(0.2),
            nn.Conv2d(64, 128, 4, 2, 1),
            nn.InstanceNorm2d(128),
            nn.LeakyReLU(0.2),
            nn.Conv2d(128, 256, 4, 2, 1),
            nn.InstanceNorm2d(256),
            nn.LeakyReLU(0.2),
            nn.Conv2d(256, 512, 4, 2, 1),
            nn.InstanceNorm2d(512),
            nn.LeakyReLU(0.2),
            nn.Conv2d(512, 1, 3, 1, 1)
        )

    def forward(self, x):
        return self.conv(x)

if __name__ == "__main__":
    img = torch.randn(2, 3, 256, 256)

    model = Generator(0.0)
    output = model(img)

    print(output.size())

    model = Discriminator()

    output = model(torch.cat([output, img], 1))

    print(output.size())

