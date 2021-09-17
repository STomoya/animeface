
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision.utils import save_image

from utils import save_args, add_args, Status
from nnutils import get_device
from nnutils.loss import gradient_penalty, NonSaturatingLoss
from tqdm import tqdm

from .model import Generator, Discriminator

def load_real(
    image_path, device,
    max_size=250, min_size=25, scale_factor=0.75, save_samples=True
):

    sizes = []
    tmp_size = max_size
    while tmp_size > min_size:
        tmp_size = round(max_size * scale_factor ** len(sizes))
        sizes.append(tmp_size)
    sizes = sorted(sizes)

    import torchvision.transforms as T
    def get_transform(size):
        return T.Compose([
            T.Resize(size),
            T.ToTensor(),
            T.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5])
        ])

    from PIL import Image
    reals = []
    xy_sizes = []
    image = Image.open(image_path).convert('RGB')
    for size in sizes:
        # prepair real image for a scale
        transform = get_transform(size)
        trans_image = transform(image)
        trans_image = trans_image.view(1, *trans_image.size())
        trans_image = trans_image.to(device)
        reals.append(trans_image)
        # get size of transformed image
        xy_sizes.append((trans_image.size(2), trans_image.size(3)))

        if save_samples:
            save_image(reals[-1], './SinGAN/result/sample_{}x{}.png'.format(*xy_sizes[-1]), normalize=True)

    return reals, xy_sizes

def test_sizes(max_size, num_scale, scale_factor, width_scale=1):
    sizes = []
    for scale in range(num_scale):
        sizes.append((round(max_size *scale_factor ** scale), round((max_size * scale_factor ** scale) * width_scale)))
    return sorted(sizes)

def train(
    epochses, G_step, D_step,
    G, D,
    reals, test,
    lr, betas,
    rec_criterion, gp_type,
    gp_lambda, rec_alpha,
    save_interval
):

    status  = Status(sum(epochses))
    loss    = NonSaturatingLoss()
    calc_gp = gradient_penalty()

    for scale, epochs in enumerate(epochses):
        tqdm.write('Scale {:2} / {:2}'.format(scale+1, len(epochses)))

        # optimizers
        optimizer_G = optim.Adam(G.parameters(), lr=lr, betas=betas)
        optimizer_D = optim.Adam(D.parameters(), lr=lr, betas=betas)
        # schedular

        for epoch in range(1, epochs+1):
            '''
            Discriminator
            '''
            for j in range(D_step):
                # D(G(z))
                fake = G.forward()
                # D(x)
                real_prob = D.forward(reals[scale])
                fake_prob = D.forward(fake.detach())

                # loss
                adv_loss = loss.d_loss(real_prob, fake_prob)
                gp_loss  = calc_gp(reals[scale], fake.detach(), D.forward, None, gp_type) * gp_lambda
                D_loss = adv_loss + gp_loss

                # optimization
                optimizer_D.zero_grad()
                D_loss.backward()
                optimizer_D.step()

            '''
            Generator
            '''
            for j in range(G_step):
                # D(G(z))
                fake = G.forward()
                rec_fake = G.forward(rec=True)
                fake_prob = D.forward(fake)

                # loss
                adv_loss = loss.g_loss(fake_prob)
                rec_loss = rec_criterion(rec_fake, reals[scale])
                # total
                G_loss = adv_loss + rec_loss * rec_alpha

                # optimization
                optimizer_G.zero_grad()
                G_loss.backward(retain_graph=True)
                optimizer_G.step()

            if status.batches_done % save_interval == 0:
                save_image(fake, f'./implementations/SinGAN/result/{scale}_{epoch}.jpg', normalize=True, value_range=(-1, 1))

            status.update(g=G_loss.item(), d=D_loss.item())

        if scale+1 < len(epochses):
            G.progress(rec_fake, reals[scale])
            D.progress()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    # G.cpu()
    if test:
        G.eval(all=True)
        img = G.forward(sizes=test)
        save_image(img, './implementations/SinGAN/result/eval_{}x{}.png'.format(*test[-1]), normalize=True, value_range=(-1, 1))

def main(parser):

    parser = add_args(parser,
        dict(
            image_path      = ['./data/animefacedataset/images/63568_2019.jpg', 'path to image'],
            max_size        = [220, 'max size when training'],
            min_size        = [25, 'min size when training'],
            scale_factor    = [0.7, 'scale factor for resizing the training image'],
            save_real       = [False, 'save real samples'],
            img_channels    = [3, 'image channels'],
            channels        = [32, 'channel width multiplier'],
            kernel_size     = [3, 'kernel size of convolution layers'],
            norm_layer      = ['bn', 'normalization layer name'],
            num_layers      = [5, 'number of layers for each scale'],
            disable_img_out = [False, 'disable Tanh on output'],
            disable_bias    = [False, 'disable bias'],
            epochs          = [3000, 'epochs to train each scale'],
            increase        = [0, 'epochs to increase in each scale'],
            G_step          = [3, 'number of G optimization steps'],
            D_step          = [3, 'number of D optimization steps'],
            lr              = [0.0005, 'learning rate'],
            betas           = [[0.5, 0.999], 'betas'],
            gp_type         = [0., 'center for gradient penalty'],
            gp_lambda       = [0.1, 'lambda for gradient penalty'],
            rec_alpha       = [10., 'alpha for reconstruction loss'],
            test_size       = [500, 'size of test image']))
    args = parser.parse_args()
    save_args(args)

    img_out = not args.disable_img_out
    bias = not args.disable_bias

    device = get_device(not args.disable_gpu)

    reals, sizes = load_real(
        args.image_path, device, args.max_size, args.min_size,
        args.scale_factor, args.save_real)
    # test = test_sizes(test_size, len(sizes), scale_factor)
    test = None

    Gs = Generator(
        sizes, device, args.img_channels, args.channels,
        args.kernel_size, args.norm_layer, args.num_layers, img_out, bias=bias)
    Ds = Discriminator(
        sizes, device, args.img_channels, args.channels,
        args.kernel_size, args.norm_layer, args.num_layers, bias=bias)
    Gs.to()
    Ds.to()

    rec_criterion = nn.MSELoss()

    train(
        [args.epochs + scale*args.increase for scale, _ in enumerate(sizes)], args.G_step, args.D_step,
        Gs, Ds,
        reals, test,
        args.lr, args.betas,
        rec_criterion, args.gp_type,
        args.gp_lambda, args.rec_alpha,
        args.save
    )
