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

import copy

import wandb

wandb.login()

os.environ['CUDA_VISIBLE_DEVICES'] = '3'

trans = transform.Compose(
    [transform.Resize((64, 64))]
)

img_dataset = ImgDataset(dir='/home/liruijun/projects/datasets/flowers/train', transform=trans)
data_loader = DataLoader(img_dataset, batch_size=128, shuffle=True)

device = torch.device('cuda:0')
lr = 1e-3
epochs = 100

model = Model()
ema_model = copy.deepcopy(model)
model = model.to(device)
ema_model = ema_model.to(device)

optimizer = Adam(model.parameters(), lr=1e-3)

loss = nn.MSELoss()

T = 500
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

image_table = wandb.Table()
for epoch in tqdm(range(epochs)):
    model.train()
    for i, imgs in enumerate(data_loader):
        optimizer.zero_grad()
        bsize = imgs.shape[0]
        for t in tqdm(range(T)):
            noise_imgs, epsilons = forward_process(imgs, t)
            pred_noise = model(noise_imgs, torch.tensor([t]*bsize, device=device).reshape((-1, 1)))
            loss_noise = loss(pred_noise, epsilons)
            loss_noise.backward()
            optimizer.step()
            update_ema_parameters()
        wandb.log({"loss": loss_noise})
    
    image_table.add_column('noise_imgs', noise_imgs)
    image_table.add_column('epsilons', epsilons)
    image_table.add_column('pred_noise', pred_noise)
    wandb.log({"images_epoch": image_table})
    

save_model_path = '/home/liruijun/projects/Generative_models/sd/trained_models'
print(f'finished training, saving model to - {save_model_path}')
torch.save(model.state_dict(), save_model_path)