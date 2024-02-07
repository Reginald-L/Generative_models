import os
import sys
sys.path.insert(0, '/home/liruijun/projects/Generative_models')
import torch
from torch import nn as nn
from model_util import *
import math
import yaml
from sd.diffusion_schedule import LinearScheduler, BaseScheduler

class ModelConfig:
    def __init__(self):
        self.model_input_size = 64
        self.num_down_blocks = 3
        self.num_up_blocks = 3
        self.img_channels = 3
        self.unet_in_channels = 64
        self.unet_out_channels = 32
        self.num_res_blocks_per_block = 2
        self.out_channels = [32, 64, 96]
        self.num_res_blocks_in_bottleneck = 2
        self.out_channels_bottleneck_res_blocks = 128


class UNet(nn.Module):
    def __init__(self, config):
        super(UNet, self).__init__()
        self.config = config
        self.encoder = self.create_encoder()
        self.bottleneck = self.create_bottleneck()
        self.decoder = self.create_decoder()
        

    def create_encoder(self):
        encoder = nn.Sequential()
        unet_in_channels = self.config.unet_in_channels
        for i in range(self.config.num_down_blocks):
            encoder.append(DownBlock(unet_in_channels, self.config.out_channels[i], self.config.num_res_blocks_per_block))
            unet_in_channels = self.config.out_channels[i]
        return encoder

    def create_decoder(self):
        decoder = nn.Sequential()
        config_out_channels = self.config.out_channels
        unet_up_in_channels = self.config.out_channels_bottleneck_res_blocks
        for i in range(self.config.num_up_blocks):
            out_channels = config_out_channels.pop()
            decoder.append(UpBlock(unet_up_in_channels + out_channels, out_channels, self.config.num_res_blocks_per_block))
            unet_up_in_channels = out_channels
        return decoder
    
    def create_bottleneck(self):
        bottleneck = nn.Sequential()
        bottleneck_in_channels = self.config.out_channels[-1]
        for i in range(self.config.num_res_blocks_in_bottleneck):
            bottleneck.append(ResBlock(bottleneck_in_channels, self.config.out_channels_bottleneck_res_blocks))
            bottleneck_in_channels = self.config.out_channels_bottleneck_res_blocks
        return bottleneck
    
    def forward(self, x):
        down_results = []
        for down_block in self.encoder:
            x, down_result = down_block(x)
            down_results.append(down_result)
        x = self.bottleneck(x)
        for up_block in self.decoder:
            x = up_block(x, down_results.pop())
        return x


class Model(nn.Module):
    def __init__(self):
        super(Model, self).__init__()
        self.config = ModelConfig()
        self.unet = UNet(self.config)
        self.noise_variance_block = nn.Upsample(scale_factor=self.config.model_input_size, mode='nearest')
        self.image_block = nn.Conv2d(self.config.img_channels, self.config.unet_in_channels // 2, kernel_size=1)
        self.convert_img_block = nn.Conv2d(self.config.unet_out_channels, self.config.img_channels, kernel_size=1)


    def sinusoidal_embedding(self, noise_variance):
        noise_variance = noise_variance.reshape((noise_variance.shape[0], 1, 1, 1))
        frequencies = torch.exp(
            torch.linspace(
                math.log(1.0),
                math.log(1000.0),
                16,
                device=noise_variance.device
            )
        )
        angular_speeds = 2.0 * math.pi * frequencies
        embeddings = torch.concat([
            torch.sin(angular_speeds * noise_variance),
            torch.cos(angular_speeds * noise_variance)
        ], dim=-1)
        return embeddings.permute(0, 3, 1, 2)
    

    def forward(self, img, noise_variance):
        noise_variance = self.sinusoidal_embedding(noise_variance)
        noise_variance_embedding = self.noise_variance_block(noise_variance)
        image_embedding = self.image_block(img)
        x = torch.cat([noise_variance_embedding, image_embedding], dim=1)
        out = self.unet(x)
        out_img = self.convert_img_block(out)
        return out_img


class GaussianDiffusion(nn.Module):
    def __init__(self, scheduler:BaseScheduler):
        super(GaussianDiffusion, self).__init__()
        self.model = Model()
        self.scheduler = scheduler
        self.noise_rates, self.signal_rates = self.scheduler.diffusion_scheduler()
        
    def forward(self, x):
        return self.model.forward(x)

    def up_dim_tensor(self, tensor, bsize):
        tensor = tensor.unsqueeze(0)
        tensor = torch.concat([tensor] * bsize, dim=0).reshape((-1, 1, 1, 1))
        return tensor

    def get_noise_signal_rates_by_t(self, t):
        noise_rate, signal_rate = self.noise_rates[t], self.signal_rates[t]
        return noise_rate, signal_rate
    
    def get_noise_signal_rates_from_custom_rates(self, noise_rates, signal_rates, t):
        print(t)
        noise_rate, signal_rate = noise_rates[t], signal_rates[t]
        return noise_rate, signal_rate

    def forward_diffusion(self, imgs, t):
        device = imgs.device
        bsize = imgs.shape[0]
        # get noise images
        # sample noise
        noise_rate, signal_rate = self.get_noise_signal_rates_by_t(t)
        noise_rate = self.up_dim_tensor(noise_rate, bsize).to(device)
        signal_rate = self.up_dim_tensor(signal_rate, bsize).to(device)
        epsilons = torch.randn_like(imgs, device=device)
        noise_imgs = signal_rate * imgs + noise_rate * epsilons
        return noise_imgs.to(device), epsilons.to(device)

    def denoise(self, noise_imgs, noise_rates, signal_rates, t):
        bsize = noise_imgs.shape[0]
        noise_rate, signal_rate = self.get_noise_signal_rates_from_custom_rates(noise_rates, signal_rates, t)
        noise_rate = self.up_dim_tensor(noise_rate, bsize)
        signal_rate = self.up_dim_tensor(signal_rate, bsize)
        pred_noises = self.model(noise_imgs, noise_rate**2)
        pred_imgs = (noise_imgs - noise_rate * pred_noises) / signal_rate
        return pred_imgs, pred_noises

    def denoise_with_model_rates(self, noise_imgs, t):
        device = noise_imgs.device
        bsize = noise_imgs.shape[0]
        noise_rate, signal_rate = self.get_noise_signal_rates_by_t(t)
        noise_rate = self.up_dim_tensor(noise_rate, bsize).to(device)
        signal_rate = self.up_dim_tensor(signal_rate, bsize).to(device)
        pred_noises = self.model(noise_imgs, noise_rate**2)
        pred_imgs = (noise_imgs - noise_rate * pred_noises) / signal_rate
        return pred_imgs.to(device), pred_noises.to(device)

    def reverse_diffusion(self, noises, diffusion_steps):
        diffusion_times = torch.tensor([x / diffusion_steps for x in range(diffusion_steps)])
        noise_rates, signal_rates = self.scheduler.diffusion_scheduler_with_diffusion_times(diffusion_times)
        current_images = noises
        for step in range(diffusion_steps)[::-1]:
            current_images, _ = self.denoise(current_images, noise_rates, signal_rates, step)
        pred_imgs = current_images
        return pred_imgs  



if __name__ == '__main__':
    # model = Model()
    # img = torch.rand(1, 3, 64, 64)
    # noise_variance = torch.tensor(1.2).reshape((-1, 1))
    # out = model(img, noise_variance)
    # print(out.shape)
    # print(model)
    T=1000
    scheduler = LinearScheduler(torch.tensor([x / T for x in range(T)]))
    diffusion = GaussianDiffusion(scheduler)
    imgs = torch.randn(size=(8, 3, 64, 64))
    diffusion.reverse_diffusion(imgs, 20)


