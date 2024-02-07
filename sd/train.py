import random
import sys
import os
sys.path.insert(0, '/home/liruijun/projects/Generative_models')
from data.data_utils import ImgDataset
from torch.utils.data import DataLoader
from torchvision import transforms as transform
import torch
from torch.optim import AdamW
from torch import nn as nn
from unet import GaussianDiffusion
from sd.diffusion_schedule import LinearScheduler
from collections import OrderedDict
from tqdm import tqdm
import numpy as np
from torchvision.utils import make_grid, save_image

import copy

import wandb

wandb.login()

trans = transform.Compose(
    [transform.Resize((64, 64))]
)

img_dataset = ImgDataset(dir='/home/liruijun/projects/datasets/flowers/train', transform=trans)
data_loader = DataLoader(img_dataset, batch_size=512, shuffle=True)


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv2d:
        nn.init.xavier_normal_(m.weight)

T = 1000

lr = 1e-4
epochs = 100
device = torch.device('cuda:1')

scheduler = LinearScheduler(torch.tensor([x / T for x in range(T)]))
model = GaussianDiffusion(scheduler)
model.apply(init_weights)
ema_model = copy.deepcopy(model)

optimizer = AdamW(model.parameters(), lr=lr)

model = model.to(device)
ema_model = ema_model.to(device)


loss = nn.MSELoss()

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

def update_ema_parameters():
    with torch.no_grad():
        model_parms = OrderedDict(model.named_parameters())
        ema_parms = OrderedDict(ema_model.named_parameters())
        assert model_parms.keys() == ema_parms.keys(), 'ema model parameters must be the same length as model parameters'
        for name, param in model_parms.items():
            ema_parms[name] = 0.999 * ema_parms[name] + (1 - 0.999) * param
        ema_model.load_state_dict(ema_parms, strict=False)


ts = [t for t in range(0, T)]
for epoch in tqdm(range(epochs)):
    model.train()
    ts = random.sample(ts, 20)
    for i, imgs in enumerate(data_loader):
        optimizer.zero_grad()
        for t in tqdm(ts):
            noise_imgs, epsilons = model.forward_diffusion(imgs.to(device), t)
            pred_imgs, pred_noise = model.denoise_with_model_rates(noise_imgs, t)
            loss_noise = torch.sqrt(loss(pred_noise, epsilons))
            loss_noise.backward()
            optimizer.step()
            update_ema_parameters()
        wandb.log({"loss": loss_noise})

        # pred_noise_imgs = imgs[:8].to('cpu') + pred_imgs[:8].to('cpu')
        pred_noise_imgs = torch.concat([imgs[:8].to('cpu'), pred_imgs[:8].to('cpu')], dim=0)
        img_grid = make_grid(pred_noise_imgs, nrow=8)
    save_image(img_grid, f'/home/liruijun/projects/Generative_models/sd/trained_models/img_{epoch}.png')

    if epoch % 5 == 0 and epoch != 0:
        save_model_path = f'/home/liruijun/projects/Generative_models/sd/trained_models/flower_{epoch}.pth'
        save_ema_model_path = f'/home/liruijun/projects/Generative_models/sd/trained_models/flower_ema_{epoch}.pth'
        print(f'saving epoch-{epoch} model to - {save_model_path}')
        torch.save(model.state_dict(), save_model_path)
        torch.save(ema_model.state_dict(), save_ema_model_path)

save_final_model_path = f'/home/liruijun/projects/Generative_models/sd/trained_models/flower.pth'
save_final_ema_model_path = f'/home/liruijun/projects/Generative_models/sd/trained_models/flower_ema.pth'
print(f'saving final model to - {save_final_model_path}')
torch.save(model.state_dict(), save_final_model_path)
torch.save(ema_model.state_dict(), save_final_ema_model_path)