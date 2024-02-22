from diffusers import AutoencoderKL
import torch
from torchvision.transforms.functional import to_tensor
from torchvision.utils import make_grid, save_image
from PIL import Image
import sys
import os
sys.path.insert(0, '/home/liruijun/projects/Generative_models')
from data.data_utils import ImgDataset
from torch.utils.data import DataLoader
from torchvision import transforms as transform
from tqdm import tqdm
from vae.vae_model import Face_Generator, Face_Encoder, Face_Decoder
import wandb

wandb.login()

trans = transform.Compose(
    [transform.Resize((64, 64))]
)

img_dataset = ImgDataset(dir='/home/liruijun/projects/datasets/img_align_celeba/img_align_celeba', transform=trans)
data_loader = DataLoader(img_dataset, batch_size=128, shuffle=True)


encoder = Face_Encoder()
decoder = Face_Decoder()
vae = Face_Generator(encoder, decoder)

reconstruction_loss = torch.nn.BCELoss(reduction='sum')
def loss(rec_x, x, mu, log_var):
    # bce loss + kl_loss
    rec_loss = reconstruction_loss(rec_x, x)
    kl_loss = -0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp())
    total_loss = rec_loss + kl_loss
    return total_loss, rec_loss, kl_loss
    # return rec_loss

lr = 1e-4
optim = torch.optim.Adam(vae.parameters(), lr=lr)

device = torch.device('cuda:0')

run = wandb.init(
    # Set the project where this run will be logged
    project="face_vae",
    # Track hyperparameters and run metadata
    config={
        "learning_rate": lr,
        "epochs": 50
    },
)

vae = vae.to(device)
vae.train()
num_show_imgs = 8
for epoch in tqdm(range(50)):
    for index, images in tqdm(enumerate(data_loader)):
        images = images.to(device)
        optim.zero_grad()
        latents, mu, log_var = vae.encoder(images)
        recon_batch = vae.decoder(latents)
        loss_value, r_loss, kl_loss = loss(recon_batch, images, mu, log_var)
        loss_value.backward()
        optim.step()
        if index % 50 == 0:
            wandb.log({'loss': loss_value.item(), 'r_loss': r_loss.item(), 'kl_loss': kl_loss.item()})
            print(f"epoch: {epoch} - iteration: {index} loss: {loss_value.item():.6f}; r_loss: {r_loss.item():.6f}; kl_loss: {kl_loss.item():.6f}")
            show_image = torch.concat([images[:num_show_imgs].to('cpu'), recon_batch[:num_show_imgs].to('cpu')], dim=0)
            wandb.log({'example_faces': [wandb.Image(img) for img in show_image]})
            img_grid = make_grid(show_image, nrow=num_show_imgs)
            save_image(img_grid, f'/home/liruijun/projects/Generative_models/vae/training_own_vae/celeba_{epoch}.png')  

    # save model
    save_model_path = f'/home/liruijun/projects/Generative_models/vae/training_own_vae/celeba_{epoch}.pth'
    print(f'saving epoch-{epoch} model to - {save_model_path}')
    torch.save(vae.state_dict(), save_model_path)