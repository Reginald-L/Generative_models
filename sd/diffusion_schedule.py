import torch
import math


class BaseScheduler():
    def __init__(self, diffusion_times):
        assert isinstance(diffusion_times, torch.Tensor), "diffusion_times must be a torch.Tensor"
        self.diffusion_times = diffusion_times

    def diffusion_scheduler(self):
        pass


class LinearScheduler(BaseScheduler):
    def __init__(self, diffusion_times):
        super(LinearScheduler, self).__init__(diffusion_times)
        self.min_rate = 0.0001
        self.max_rate = 0.02

    def diffusion_scheduler(self):
        betas = self.min_rate + self.diffusion_times * (self.max_rate - self.min_rate)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        signal_rates = torch.sqrt(alpha_bars)
        noise_rates = torch.sqrt(1 - alpha_bars)
        return noise_rates, signal_rates
    
    def diffusion_scheduler_with_diffusion_times(self, diffusion_times):
        betas = self.min_rate + diffusion_times * (self.max_rate - self.min_rate)
        alphas = 1 - betas
        alpha_bars = torch.cumprod(alphas, dim=0)
        signal_rates = torch.sqrt(alpha_bars)
        noise_rates = torch.sqrt(1 - alpha_bars)
        return alphas, alpha_bars, noise_rates, signal_rates


class CosineScheduler(BaseScheduler):
    def __init__(self, diffusion_times):
        super(CosineScheduler, self).__init__(diffusion_times)
    
    def diffusion_scheduler(self):
        signal_rates = torch.cos(self.diffusion_times * int(torch.pi / 2))
        noise_rates = torch.sin(self.diffusion_times * int(torch.pi / 2))
        return noise_rates, signal_rates


class CosineOffsetScheduler(BaseScheduler):
    def __init__(self, diffusion_times):
        super(CosineOffsetScheduler, self).__init__(diffusion_times)
        self.min_signal_rate = torch.tensor(0.02)
        self.max_signal_rate = torch.tensor(0.95)
    
    def diffusion_scheduler(self):
        start_angle = torch.acos(self.max_signal_rate)
        end_angle = torch.acos(self.min_signal_rate)
        diffusion_angles = start_angle + self.diffusion_times * (end_angle - start_angle)
        signal_rates = torch.cos(diffusion_angles)
        noise_rates = torch.sin(diffusion_angles)
        return noise_rates, signal_rates


if __name__ == '__main__':
    import os
    os.environ['CUDA_VISIBLE_DEVICES'] = '3'
    T = 200
    diffusion_times = [x / T for x in range(T)]
    scheduler = LinearScheduler(diffusion_times)
    noise_rates, signal_rates = scheduler.diffusion_scheduler()
    print(noise_rates.shape)
    print(signal_rates.shape)

    # scheduler = CosineScheduler(diffusion_times)
    # noise_rates, signal_rates = scheduler.diffusion_scheduler()
    # print(noise_rates.shape)
    # print(signal_rates.shape)

    # scheduler = CosineOffsetScheduler(diffusion_times)
    # noise_rates, signal_rates = scheduler.diffusion_scheduler()
    # print(noise_rates.shape)
    # print(signal_rates.shape)
    from PIL import Image
    from torchvision.transforms import ToTensor, ToPILImage
    from torchvision.utils import make_grid
    import numpy as np
    img = Image.open('/home/liruijun/projects/scripts/test/Meihuo_test/Meihuo_test/human/Model_08.jpg').convert('RGB')

    img_tensor = ToTensor()(img)
    imgs = []
    for i in range(T):
        if i % 10 == 0:
            epsilon = torch.randn_like(img_tensor)
            img = signal_rates[i] * img_tensor + noise_rates[i] * epsilon
            img = ToPILImage()(img)
            # imgs.append(img)
            img.save(f'./imgs_{i}.png')
        
    # imgs = np.stack(imgs, axis=-1)
    # grid = Image.fromarray(imgs)
    # grid.save('./imgs.png')
