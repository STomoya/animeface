var app = new Vue({
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
                {name: "StyleGAN2", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/StyleGAN2/result/118000.png"},
                {name: "StyleGAN2 + DiffAugment", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/DiffAugment/result/238000.png"}
            ],
            [
                {name: "BigGAN (failed)", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/BigGAN/result/14000.png"},
                {name: "StyleGAN2 + AdaBelief + DiffAugment", src: "https://raw.githubusercontent.com/STomoya/animeface/master/implementations/AdaBelief/result/108000.png"},
                {name: "", src: ""}
            ]
        ]
    }
})