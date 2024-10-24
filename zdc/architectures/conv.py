from typing import List

import jax.numpy as jnp
from flax import linen as nn

from zdc.layers import UpSample


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
                x = Conv(channels, kernel_size=3, strides=2)(x)

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
                x = Conv(channels, kernel_size=3)(x)

        x = LayerNormF32()(x)
        x = nn.swish(x)
        x = Conv(1, kernel_size=1)(x)
        x = nn.relu(x)

        return x


class AttentionBlock(nn.Module):
    channels: int
    n_heads: int

    @nn.compact
    def __call__(self, x):
        residual = x

        x = LayerNormF32()(x)
        x = nn.MultiHeadDotProductAttention(
            num_heads=self.n_heads,
            qkv_features=self.channels,
            dtype=jnp.bfloat16,
            out_kernel_init=nn.initializers.normal(0.2 / self.channels ** 0.5),
            use_bias=False
        )(x)
        x = x + residual

        return x


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        residual = x

        x = LayerNormF32()(x)
        x = nn.swish(x)
        x = Conv(self.channels, kernel_size=3)(x)

        x = LayerNormF32()(x)
        x = nn.swish(x)
        x = Conv(self.channels, kernel_size=3, init_std=0.0001 / self.channels)(x)

        if residual.shape[-1] != self.channels:
            residual = Conv(self.channels, kernel_size=1)(residual)

        return x + residual


class LayerNormF32(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = x.astype(jnp.float32)
        x = nn.LayerNorm(epsilon=1e-6)(x)
        return x.astype(x.dtype)


class Conv(nn.Module):
    channels: int
    kernel_size: int
    strides: int = 1
    init_std: float = None

    @nn.compact
    def __call__(self, x):
        if self.init_std is not None:
            init_std = self.init_std
        else:
            init_std = 1.0 / (self.channels * self.kernel_size ** 2)

        return nn.Conv(
            features=self.channels,
            kernel_size=(self.kernel_size, self.kernel_size),
            strides=(self.strides, self.strides),
            padding='SAME',
            use_bias=False,
            dtype=jnp.bfloat16,
            kernel_init=nn.initializers.normal(init_std, dtype=jnp.float32)
        )(x)
