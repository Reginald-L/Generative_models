import random
import sys
import os
sys.path.insert(0, '/home/liruijun/projects/Generative_models')
from data.data_utils import ImgDataset
from torch.utils.data import DataLoader
from torchvision import transforms as transform
import torch
from torch.optim import Adam
from torch import nn as nn
from unet import Model
from sd.diffusion_schedule import LinearScheduler
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from torchvision.utils import make_grid, save_image

import copy

import wandb

wandb.login()

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

trans = transform.Compose(
    [transform.Resize((64, 64))]
)

img_dataset = ImgDataset(dir='/home/liruijun/projects/datasets/flowers/train', transform=trans)
data_loader = DataLoader(img_dataset, batch_size=128, shuffle=True)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)

device = torch.device('cuda:0')
lr = 1e-3
epochs = 200

model = Model()
model.apply(init_weights)
ema_model = copy.deepcopy(model)
model = model.to(device)
ema_model = ema_model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

loss = nn.MSELoss()

T = 1000
diffusion_times = [x / T for x in range(T)]
scheduler = LinearScheduler(diffusion_times)
noise_rates, signal_rates = scheduler.diffusion_scheduler()

run = wandb.init(
    # Set the project where this run will be logged
    project="ddpm",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": epochs,
        "T": T
    },
)

def forward_process(images, t):
    noise_images = []
    epsilons = []
    for image in images:
        epsilon = torch.randn_like(image)
        noise_image = signal_rates[t] * image + noise_rates[t] * epsilon
        noise_images.append(noise_image.unsqueeze(0))
        epsilons.append(epsilon.unsqueeze(0))
    return torch.concat(noise_images, dim=0).to(device), torch.concat(epsilons, dim=0).to(device)


def update_ema_parameters():
    model_parms = OrderedDict(model.named_parameters())
    ema_parms = OrderedDict(ema_model.named_parameters())
    assert model_parms.keys() == ema_parms.keys(), 'ema model parameters must be the same length as model parameters'
    for name, param in model_parms.items():
        ema_parms[name] = 0.999 * ema_parms[name] + (1 - 0.999) * param
    ema_model.load_state_dict(ema_parms, strict=False)

ts = [t for t in range(0, T)]
for epoch in tqdm(range(epochs)):
    model.train()
    ts = random.sample(ts, 10)
    for i, imgs in enumerate(data_loader):
        optimizer.zero_grad()
        bsize = imgs.shape[0]
        for t in tqdm(ts):
            noise_imgs, epsilons = forward_process(imgs, t)
            pred_noise = model(noise_imgs, torch.tensor([t]*bsize, device=device).reshape((-1, 1)))
            loss_noise = loss(pred_noise, epsilons)
            loss_noise.backward()
            optimizer.step()
            update_ema_parameters()
        wandb.log({"loss": loss_noise})

        pred_noise_imgs = pred_noise[:8]
        img_grid = make_grid(pred_noise_imgs, nrow=8)
    save_image(img_grid, f'/home/liruijun/projects/Generative_models/sd/trained_models/img_{epoch}.png')

    
save_model_path = '/home/liruijun/projects/Generative_models/sd/trained_models/flower.pth'
print(f'finished training, saving model to - {save_model_path}')
torch.save(model.state_dict(), save_model_path)