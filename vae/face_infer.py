from diffusers import AutoencoderKL
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid, save_image
from PIL import Image
import sys
sys.path.insert(0, '/home/liruijun/projects/Generative_models')
from vae.vae_model import Face_Generator, Face_Encoder, Face_Decoder

# url = 'https://huggingface.co/stabilityai/sd-vae-ft-mse-original/blob/main/vae-ft-mse-840000-ema-pruned.safetensors'
# vae = AutoencoderKL.from_single_file(url)

encoder = Face_Encoder()
decoder = Face_Decoder()
vae = Face_Generator(encoder, decoder)

# pth_path = '/home/liruijun/projects/Generative_models/vae/trained_models/celeba_25.pth'
pth_path = '/home/liruijun/projects/Generative_models/vae/training_own_vae/celeba_49.pth'
vae.load_state_dict(torch.load(pth_path))

device = torch.device('cuda:0')
vae = vae.to(device)
vae.eval()

samples = torch.randn(size=(1, 256), device=device)
results = vae.decoder(samples)

grid = make_grid(torch.clip(results * 0.5 + 0.5, 0, 1), nrow=1)
save_image(grid, '/home/liruijun/projects/Generative_models/vae/vae_samples.png')