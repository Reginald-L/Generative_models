import sys
import os
sys.path.insert(0, '/home/liruijun/projects/Generative_models')
import torch
from sd.unet import GaussianDiffusion
from sd.diffusion_schedule import LinearScheduler

from torchvision.utils import make_grid, save_image

os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,2,3'

device = torch.device('cuda:1')

diffusion_steps = 20

schedure = LinearScheduler(torch.tensor([x / diffusion_steps for x in range(diffusion_steps)]))
model = GaussianDiffusion(schedure)

ckpt_path = '/home/liruijun/projects/Generative_models/sd/trained_models/flower_95_64_mse.pth'
model.load_state_dict(torch.load(ckpt_path))
model = model.to(device)
model.eval()

noises = torch.randn((1, 3, 64, 64), device=device)
pred_imgs = model.reverse_diffusion(noises, diffusion_steps)

grid = make_grid(torch.clip(pred_imgs * 0.5 + 0.5, 0, 1), nrow=1)
save_image(grid, f'/home/liruijun/projects/Generative_models/sd/trained_models/imgs.png')