from typing import List

from flax import linen as nn

from zdc.layers import AttentionBlock, Conv, DownSample, LayerNormF32, ResidualBlock, UpSample


class Encoder(nn.Module):
    channels: int
    channel_multipliers: List[int]
    n_resnet_blocks: int
    n_heads: int

    @nn.compact
    def __call__(self, img):
        x = Conv(self.channels, kernel_size=3)(img)

        for i, multiplier in enumerate(self.channel_multipliers):
            channels = self.channels * multiplier
            for _ in range(self.n_resnet_blocks):
                x = ResidualBlock(channels)(x)
            if i != len(self.channel_multipliers) - 1:
                x = DownSample()(x)

        x = ResidualBlock(channels)(x)
        x = AttentionBlock(channels, self.n_heads)(x)
        x = ResidualBlock(channels)(x)

        x = LayerNormF32()(x)
        x = nn.swish(x)

        return x


class Decoder(nn.Module):
    channels: int
    channel_multipliers: List[int]
    n_resnet_blocks: int
    n_heads: int

    @nn.compact
    def __call__(self, z):
        channels = self.channels * self.channel_multipliers[-1]
        x = Conv(channels, kernel_size=3)(z)

        x = ResidualBlock(channels)(x)
        x = AttentionBlock(channels, self.n_heads)(x)
        x = ResidualBlock(channels)(x)

        for i, multiplier in enumerate(self.channel_multipliers[::-1]):
            channels = self.channels * multiplier
            for _ in range(self.n_resnet_blocks):
                x = ResidualBlock(channels)(x)
            if i != 0:
                x = UpSample()(x)

        x = LayerNormF32()(x)
        x = nn.swish(x)
        x = Conv(1, kernel_size=1)(x)
        x = nn.relu(x)

        return x
