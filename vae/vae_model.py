import torch
from torch import nn as nn


class Face_Generator(nn.Module):
    def __init__(self, encoder, decoder) -> None:
        super(Face_Generator, self).__init__()
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, x):
        latent_samples, mu, log_var = self.encoder(x)
        output = self.decoder(latent_samples)
        return output, mu, log_var
    
class Face_Encoder(nn.Module): 
    def __init__(self) -> None:
        super(Face_Encoder, self).__init__()
        self.model_util = Model_utils()
        self.encode_block1 = self.model_util.conv_bn_lrelu(3, 32, ksize=3, stride=2, padding=1)
        self.encode_block1_1 = self.model_util.conv_bn_lrelu(32, 32, ksize=3, stride=1, padding=1)
        self.encode_block2 = self.model_util.conv_bn_lrelu(32, 64, ksize=3, stride=2, padding=1)
        self.encode_block2_1 = self.model_util.conv_bn_lrelu(64, 64, ksize=3, stride=1, padding=1)
        self.encode_block3 = self.model_util.conv_bn_lrelu(64, 128, ksize=3, stride=2, padding=1)
        # flatten
        self.flatten = nn.Flatten()
        # mu
        self.mu_layer = nn.Linear(128 * 8 * 8, 256)
        # log_var
        self.log_var_layer = nn.Linear(128 * 8 * 8, 256)

    # sample
    def sample_z(self, mu, log_var):
        epsilon = torch.randn_like(mu, device=mu.device)
        return mu + torch.exp(log_var / 2) * epsilon

    def forward(self, x):
        x = self.encode_block1(x)
        x = self.encode_block1_1(x)
        x = self.encode_block2(x)
        x = self.encode_block2_1(x)
        x = self.encode_block3(x)
        # flatten
        x_flatten = self.flatten(x)
        # mu
        mu = self.mu_layer(x_flatten)
        # log_var
        log_var = self.log_var_layer(x_flatten)
        z = self.sample_z(mu, log_var)
        return z, mu, log_var


class Face_Decoder(nn.Module):
    def __init__(self) -> None:
        super(Face_Decoder, self).__init__()
        self.model_util = Model_utils()
        self.input_process_layer = nn.Linear(256, 128 * 8 * 8)
        self.decode_block1 = self.model_util.deconv_bn_lrelu(128, 64, ksize=4, stride=2, padding=1)
        self.decode_block1_1 = self.model_util.deconv_bn_lrelu(64, 64, ksize=3, stride=1, padding=1)
        self.decode_block2 = self.model_util.deconv_bn_lrelu(64, 32, ksize=4, stride=2, padding=1)
        self.decode_block2_1 = self.model_util.deconv_bn_lrelu(32, 32, ksize=3, stride=1, padding=1)
        self.decode_block3 = self.model_util.deconv_bn_sigmal(32, 3, ksize=4, stride=2, padding=1)


    def forward(self, x):
        x = self.input_process_layer(x)
        x = x.reshape((-1, 128, 8, 8))
        x = self.decode_block1(x)
        x = self.decode_block1_1(x)
        x = self.decode_block2(x)
        x = self.decode_block2_1(x)
        x = self.decode_block3(x)
        return x



class Model_utils(nn.Module):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def conv_bn_lrelu(self, in_c, out_c, ksize=3, stride=1, padding=1):
        layers = nn.Sequential(
            nn.Conv2d(in_channels=in_c, out_channels=out_c, kernel_size=ksize, stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU()
        )
        return layers    
    
    def deconv_bn_lrelu(self, in_c, out_c, ksize=3, stride=1, padding=1):
        layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=ksize, stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.LeakyReLU()
        )
        return layers 
    
    def deconv_bn_sigmal(self, in_c, out_c, ksize=3, stride=1, padding=1):
        layers = nn.Sequential(
            nn.ConvTranspose2d(in_channels=in_c, out_channels=out_c, kernel_size=ksize, stride=stride, padding=padding),
            nn.BatchNorm2d(out_c),
            nn.Sigmoid()
        )
        return layers 

if __name__ == '__main__':
    encoder = Face_Encoder()
    inputs = torch.randn((1, 3, 64, 64))
    # outputs, mu, log_var, shape = encoder(inputs)

    decoder = Face_Decoder()
    # outputs = decoder(outputs)

    vae = Face_Generator(encoder, decoder)
    out, mu, log_var = vae(inputs)
    print(out.shape)
