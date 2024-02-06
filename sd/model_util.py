import torch
from torch import nn as nn


class ResBlock(nn.Module):
    def __init__(self, in_channel, out_channel, stride=1):
        super(ResBlock, self).__init__()
        assert stride in (1, 2)
        if stride ==2 or in_channel != out_channel:
            # 输入通道和输出通道不一致,使用核大小为1的卷积层作为skip connection
            self.skip = nn.Conv2d(in_channel, out_channel, 1, stride)
        else:
            # 输入通道和输出通道一致
            self.skip = nn.Identity()

        # main path
        self.conv = nn.Sequential(
            nn.ReLU(),
            nn.Conv2d(in_channel, out_channel, 3, stride, padding=1),
            nn.ReLU(),
            nn.Conv2d(out_channel, out_channel, 3, 1, padding=1)
        )
    
    def forward(self, x):
        x1 = self.skip(x)
        x2 = self.conv(x)
        return x1 + x2


class DownBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_resblocks):
        super(DownBlock, self).__init__()
        self.residual_blocks = nn.Sequential()
        for _ in range(n_resblocks):
            self.residual_blocks.append(
                ResBlock(in_channel, out_channel, stride=1)
            )
            in_channel = out_channel
        self.avepool = nn.AvgPool2d(2, 2)

    def forward(self, x):
        res_results = []
        for res_block in self.residual_blocks:
            x = res_block(x)
            res_results.append(x)
        x = self.avepool(x)
        return x, res_results


class UpBlock(nn.Module):
    def __init__(self, in_channel, out_channel, n_resblocks):
        super(UpBlock, self).__init__()
        self.upsample = nn.Upsample(scale_factor=2, mode='bilinear')
        self.residual_blocks = nn.Sequential()
        for _ in range(n_resblocks):
            self.residual_blocks.append(
                ResBlock(in_channel, out_channel, stride=1)
            )
            in_channel = out_channel * 2
    
    def reduce_channels(self, in_channel, out_channel, x):
        reduce_channel_conv = nn.Conv2d(in_channel, out_channel, kernel_size=1)
        return reduce_channel_conv(x)
        

    def forward(self, x, down_residual_results):
        x = self.upsample(x)
        before_concat_tensor = x
        for res_block in self.residual_blocks:
            down_residual_result = down_residual_results.pop()
            after_concat_tensor = torch.cat([before_concat_tensor, down_residual_result], dim=1)
            after_resblock = res_block(after_concat_tensor)
            before_concat_tensor = after_resblock
        return after_resblock


if __name__ == '__main__':
    x = torch.rand(1, 3, 512, 512)
    # resblock = ResBlock(3, 3)
    # y = resblock(x)
    # print(y.shape)

    downblock = DownBlock(3, 64, 2)
    y, results = downblock(x)
    # print(y.shape)
    # print((results[0].shape))

    upblock = UpBlock(64, 3, 2)
    y = upblock(y, results)
    print(y.shape)