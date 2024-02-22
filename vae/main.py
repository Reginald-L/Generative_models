from diffusers import AutoencoderKL
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid, save_image
from PIL import Image

url = 'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors'
vae = AutoencoderKL.from_single_file(url)

# url = '/home/liruijun/projects/pre_trained_models/stable-diffusion-v1-5/vae'
# vae = AutoencoderKL.from_pretrained(url)

# encoder = vae.encoder
# decoder = vae.decoder

img1 = Image.open('/home/liruijun/projects/Generative_models/vae/Model_08_face.jpg')
img2 = Image.open('/home/liruijun/projects/Generative_models/vae/Model_02_face.jpg')
img_tensor1 = to_tensor(img1).unsqueeze(0)
img_tensor2 = to_tensor(img2).unsqueeze(0)
img_tensor = torch.concat([img_tensor1, img_tensor2], dim=0)
print(img_tensor.shape)

latent = vae.encode(img_tensor)
latent_tuple = latent.to_tuple()
print(len(latent_tuple))

face_distribution = latent_tuple[0]
print(type(face_distribution)) # <class 'diffusers.models.autoencoders.vae.DiagonalGaussianDistribution'>

mean, std = face_distribution.mean, face_distribution.std
print(f'mean: {mean.shape}, std: {std.shape}')
# kl = face_distribution.kl()
# print(kl)

# mode = face_distribution.mode()
# print(mode)

mean = torch.mean(mean, dim=0)
std = torch.mean(std, dim=0)

generator = torch.Generator().manual_seed(200)
esp = torch.randn((1, 4, 64, 64), generator=generator)
sample = mean + std * esp
print(sample.shape)
# esp = torch.randn_like(sample)
# # sample = sample + esp
# sample = torch.randn((4, 4, 64, 64))
out = vae.decode(sample).sample

grid_imgs = make_grid(out, nrow=1)
save_image(grid_imgs, '/home/liruijun/projects/Generative_models/vae/out.png')