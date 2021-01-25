var results = new Vue({
    el: "#results",
    data: {
        results: [
            [
                {name: "GAN", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/GAN/result/9000.png"},
                {name: "DCGAN", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/DCGAN/result/50000.png"},
                {name: "cGAN", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/cGAN/result/year_108000.png"}
            ],
            [
                {name: "ACGAN", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/ACGAN/result/i2v_118000.png"},
                {name: "pixelshuffle", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/pixelshuffle/result/i2v_119000.png"},
                {name: "pix2pix", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/pix2pix/result/155000.png"}
            ],
            [
                {name: "WGAN", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/WGAN/result/100000.png"},
                {name: "WGAN-gp", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/WGAN_gp/result/140500.png"},
                {name: "PGGAN", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/PGGAN/result/160000_wgangp.png"}
            ],
            [
                {name: "StyleGAN", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/StyleGAN/result/155000.png"},
                {name: "StyleGAN2", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/StyleGAN2/result/79000.png"},
                {name: "StyleGAN2 + DiffAugment", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/DiffAugment/result/238000.png"}
            ],
            [
                {name: "BigGAN (failed)", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/BigGAN/result/14000.png"},
                {name: "StyleGAN2 + AdaBelief + DiffAugment", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/AdaBelief/result/108000.png"},
                {name: "pix2pixHD", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/pix2pixHD/result/220000.jpg"}
            ],
            [
                {name: "SPADE", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/SPADE/result/test_185000.jpg"},
                {name: "CycleGAN", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/CycleGAN/result/test_416000.jpg"},
                {name: "", src: ""}
            ]
        ]
    }
})

var weights = new Vue({
    el: "#weights",
    data: {
        license: "https://github.com/STomoya/animeface/blob/master/LICENSE",
        weights: [
            {
                name: "StyleGAN2 animeface 128pix",
                link: "https://drive.google.com/file/d/1TkeD5oNK8hzyfY3dcf8W_gdi2ZVELEoK/view?usp=sharing",
                src:  "https://github.com/STomoya/animeface/blob/master/implementations/StyleGAN2/model.py",
                code: {
                    param: "G = Generator(\n  image_size=128, image_channles=3, style_dim=512,\n  channels=32, max_channels=512, block_num_conv=2,\n  map_num_layers=8, map_lr=0.01)",
                    input: "z = torch.empty(1, 512).normal_(0., 1.)"
                }
            }
        ]
    }
})