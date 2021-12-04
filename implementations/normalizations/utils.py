
import torch
import torch.nn as nn
import torch.optim as optim
from torch.cuda.amp import autocast, GradScaler
from torchvision.utils import save_image
import torchvision.transforms as T

from utils import Status, add_args, save_args, make_image_grid
from nnutils import get_device, init
from nnutils.loss import LSGANLoss, VGGLoss
from thirdparty.diffaugment import DiffAugment

from .data import AnimeFace, DanbooruPortraitsTest
from .model import Generator, MultiScale

def train(args,
    max_iters, dataset, test_set,
    G, D, optimizer_G, optimizer_D,
    content_lambda, style_lambda, recon_lambda,
    device, amp, save=1000,
    log_file='log.log', log_interval=10
):

    status = Status(
        max_iters, log_file is None,
        log_file, log_interval, __name__)

    adv_fn = LSGANLoss()
    vgg_fn = VGGLoss(device, p=1)
    mse_fn = nn.MSELoss()
    l1_fn  = nn.L1Loss()

    scaler = GradScaler() if amp else None
    half_way = max_iters // 2
    lr_delta = optimizer_G.param_groups[0]['lr'] / (half_way - 1)

    status.log_training(args, G, D)

    while not status.is_end():
        for content, style in dataset:
            optimizer_D.zero_grad()
            optimizer_G.zero_grad()

            content = content.to(device)
            style   = style.to(device)

            # generate
            with autocast(amp):
                style = DiffAugment(style, 'color')
                fake = G(content, style)

            '''Discriminator'''
            with autocast(amp):
                # D(real)
                real_probs = D(style)
                # D(G(c, s))
                fake_probs = D(fake.detach())

                # loss
                D_loss = sum([
                    adv_fn.d_loss(real_prob, fake_prob)
                    for real_prob, fake_prob in zip(real_probs, fake_probs)])

            if scaler is not None:
                scaler.scale(D_loss).backward()
                scaler.step(optimizer_D)
            else:
                D_loss.backward()
                optimizer_D.step()

            with autocast(amp):
                # D(G(c, s))
                fake_probs = D(fake)

                # loss
                adv_loss = sum([
                    adv_fn.g_loss(fake_prob)
                    for fake_prob in fake_probs])
                recon_loss = l1_fn(fake, style) * recon_lambda
                style_loss = vgg_fn.style_loss(style, fake) * style_lambda
                content_loss = vgg_fn.content_loss(
                    content.repeat(1, 3, 1, 1), fake) * content_lambda
                G_loss = adv_loss + recon_loss + style_loss + content_loss

            if scaler is not None:
                scaler.scale(G_loss).backward()
                scaler.step(optimizer_G)
            else:
                G_loss.backward()
                optimizer_G.step()

            if status.batches_done % save == 0:
                images, nrow = translate(G, test_set, device)
                save_image(
                    images, f'implementations/normalizations/result/{status.batches_done}.jpg',
                    nrow=nrow, normalize=True, value_range=(-1, 1))
            save_image(make_image_grid(content.repeat(1, 3, 1, 1), style, fake),
                f'running.jpg', nrow=3*3, normalize=True, value_range=(-1, 1))

            status.update(G=G_loss.item(), D=D_loss.item())
            if scaler is not None:
                scaler.update()
            G.post_step()

            if status.batches_done > half_way:
                # linearly decay lr to 0 like in pix2pix etc.
                optimizer_G.param_groups[0]['lr'] -= lr_delta
                optimizer_D.param_groups[0]['lr'] -= lr_delta

            if status.is_end():
                break
    status.plot_loss()

@torch.no_grad()
def translate(G, test_set, device):
    G.eval()
    contents, references = test_set
    # top row [empty,refs...]
    _row = [torch.zeros(references[0].size(), device=device)] + list(references)
    nrow = len(_row)
    images = [torch.cat(_row, dim=0)]
    for content in contents:
        # other rows [content, generated...]
        _row = [content.repeat(1, 3, 1, 1)]
        for reference in references:
            _row.append(
                G(content, reference))
        images.append(torch.cat(_row, dim=0))
    G.train()
    return torch.cat(images, dim=0), nrow

def main(parser):

    parser = add_args(parser,
        dict(
            num_test       = [4, 'number images for test.'],
            norm_name      = ['in', 'normalization layer name'],
            lr             = [0.0002, 'learning rate'],
            betas          = [[0.5, 0.999], 'betas'],
            style_lambda   = [10., 'lambda for style loss'],
            content_lambda = [0., 'lambda for content loss'],
            recon_lambda   = [5., 'lambda for reconstruction loss']
        ))
    args = parser.parse_args()
    save_args(args)

    device = get_device(not args.disable_gpu)
    amp    = not args.disable_gpu and not args.disable_amp

    # dataset
    # train
    dataset = AnimeFace.asloader(
        args.batch_size, (args.image_size, ),
        pin_memory=not args.disable_gpu)
    # test
    test = DanbooruPortraitsTest.asloader(
        args.num_test, (args.image_size, args.num_test),
        shuffle=False, pin_memory=False)
    # (content tensor, style tensor, t(real), real)
    test_set = next(iter(test))
    # ((contents), (styles))
    test_set = tuple(map(lambda x:x.to(device).chunk(x.size(0), dim=0), test_set))

    if args.max_iters < 0:
        args.max_iters = len(dataset) * args.default_epochs

    # model
    # only normalization layer name as a controllable parameter
    G = Generator(
        args.norm_name, content_channels=1, style_channels=3,
        content_dim=512, style_dim=512, num_resblocks=4,
        channels=32, max_channels=512, image_size=args.image_size, bottom=16)
    D = MultiScale(
        num_discs=1, in_channels=3, channels=64, max_channels=512,
        num_layers=4, image_size=args.image_size)
    # G.apply(init().N002)
    # D.apply(init().N002)
    G.to(device)
    D.to(device)

    # optimizer
    optimizer_G = optim.Adam(G.parameters(), lr=args.lr, betas=args.betas)
    optimizer_D = optim.Adam(D.parameters(), lr=args.lr, betas=args.betas)

    train(args,
        args.max_iters, dataset, test_set,
        G, D, optimizer_G, optimizer_D,
        args.content_lambda, args.style_lambda, args.recon_lambda,
        device, amp, args.save)
