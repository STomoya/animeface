
# Pre-trained weights

<details><summary>StyleGAN2 animeface 128pix</summary><div>

Download the weights and or copy-and-paste the model from here : 
[weights](https://drive.google.com/file/d/1TkeD5oNK8hzyfY3dcf8W_gdi2ZVELEoK/view?usp=sharing) | 
[model](https://github.com/STomoya/animeface/blob/master/implementations/StyleGAN2/model.py)

- Model parameters and weight loading.

    ```python
    G = Generator(
        image_size=128, image_channles=3, style_dim=512,
        channels=32, max_channels=512, block_num_conv=2,
        map_num_layers=8, map_lr=0.01
    )

    state_dict = torch.load('StyleGAN2_animeface_128pix.pt')
    G.load_state_dict(state_dict)
    ```

- Input noise sampler

    ```python
    def sampler(num_image):
        return torch.randn(num_image, 512)
    ```

- Generate image

    ```python
    num_images = 1
    images = G(sampler(num_images))
    ```

- Style mixing

    ```python
    num_images = 1
    input = (sampler(num_images), sampler(num_images))
    # 0 <= injection < G.synthesis.num_layers
    images = G(input, injection=4)
    ```

- Use `torchvision.utils.save_image` to save the images.

</div></details>
