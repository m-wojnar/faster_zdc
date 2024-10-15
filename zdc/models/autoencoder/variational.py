from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Reshape, Sampling, UpSample
from zdc.utils.data import load
from zdc.utils.losses import kl_loss, mse_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


optimizer = opt_with_cosine_schedule(
    optimizer=partial(optax.adam),
    peak_value=3e-4,
    epochs=100,
    batch_size=256
)


class VAE(nn.Module):
    channels: int = 4
    z_channels: int = 8
    channel_multipliers: tuple = (1, 2, 4)
    n_resnet_blocks: int = 1

    @nn.compact
    def __call__(self, img):
        z, z_mean, z_log_var = self.encode(img)
        return self.decode(z), z_mean, z_log_var

    def encode(self, img):
        z = Encoder(self.channels, self.channel_multipliers, self.n_resnet_blocks)(img)
        z_mean = nn.Conv(self.z_channels, kernel_size=(1, 1), dtype=jnp.bfloat16)(z)
        z_log_var = nn.Conv(self.z_channels, kernel_size=(1, 1, 1), dtype=jnp.bfloat16)(z)
        return Sampling()(z_mean, z_log_var), z_mean, z_log_var

    def decode(self, z):
        return Decoder(self.channels, self.channel_multipliers, self.n_resnet_blocks)(z)

    def gen(self, z):
        return self.decode(z)


class Encoder(nn.Module):
    channels: int
    channel_multipliers: List[int]
    n_resnet_blocks: int

    @nn.compact
    def __call__(self, img):
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding='SAME', dtype=jnp.bfloat16)(img)

        for i, multiplier in enumerate(self.channel_multipliers):
            channels = self.channels * multiplier
            for _ in range(self.n_resnet_blocks):
                x = ResidualBlock(channels)(x)
            if i != len(self.channel_multipliers) - 1:
                x = nn.Conv(channels, kernel_size=(3, 3), strides=(2, 2), padding='SAME', dtype=jnp.bfloat16)(x)

        x = ResidualBlock(channels)(x)
        x = AttentionBlock(channels)(x)
        x = ResidualBlock(channels)(x)

        x = nn.LayerNorm(dtype=jnp.bfloat16)(x)
        x = nn.swish(x)

        return x


class Decoder(nn.Module):
    channels: int
    channel_multipliers: List[int]
    n_resnet_blocks: int

    @nn.compact
    def __call__(self, z):
        channels = self.channels * self.channel_multipliers[-1]
        x = nn.Conv(channels, kernel_size=(3, 3), padding='SAME', dtype=jnp.bfloat16)(z)

        x = ResidualBlock(channels)(x)
        x = AttentionBlock(channels)(x)
        x = ResidualBlock(channels)(x)

        for i, multiplier in enumerate(self.channel_multipliers[::-1]):
            channels = self.channels * multiplier
            for _ in range(self.n_resnet_blocks):
                x = ResidualBlock(channels)(x)
            if i != 0:
                x = UpSample()(x)
                x = nn.Conv(channels, kernel_size=(3, 3), padding='SAME', dtype=jnp.bfloat16)(x)

        x = nn.LayerNorm(dtype=jnp.bfloat16)(x)
        x = nn.swish(x)
        x = nn.Conv(1, kernel_size=(3, 3), padding='SAME', dtype=jnp.bfloat16)(x)

        return x


class AttentionBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        residual = x

        x = nn.LayerNorm(dtype=jnp.bfloat16)(x)
        q = nn.Conv(self.channels, kernel_size=(1, 1), dtype=jnp.bfloat16)(x)
        k = nn.Conv(self.channels, kernel_size=(1, 1), dtype=jnp.bfloat16)(x)
        v = nn.Conv(self.channels, kernel_size=(1, 1), dtype=jnp.bfloat16)(x)

        _, h, w, c = q.shape
        q = Reshape((h * w, c))(q)
        k = Reshape((h * w, c))(k)
        v = Reshape((h * w, c))(v)

        x = jnp.einsum('bci,bcj->bij', q, k) / jnp.sqrt(self.channels)
        x = nn.softmax(x, axis=-1)
        x = jnp.einsum('bij,bcj->bci', x, v)

        x = Reshape((h, w, c))(x)
        x = nn.Conv(self.channels, kernel_size=(1, 1), dtype=jnp.bfloat16)(x)
        x = x + residual

        return x


class ResidualBlock(nn.Module):
    channels: int

    @nn.compact
    def __call__(self, x):
        residual = x

        x = nn.LayerNorm(dtype=jnp.bfloat16)(x)
        x = nn.swish(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding='SAME', dtype=jnp.bfloat16)(x)

        x = nn.LayerNorm(dtype=jnp.bfloat16)(x)
        x = nn.swish(x)
        x = nn.Conv(self.channels, kernel_size=(3, 3), padding='SAME', dtype=jnp.bfloat16)(x)

        if residual.shape[-1] != self.channels:
            residual = nn.Conv(self.channels, kernel_size=(1, 1), dtype=jnp.bfloat16)(residual)

        return x + residual


def loss_fn(params, state, key, img, cond, model, kl_weight=0.7):
    (reconstructed, z_mean, z_log_var), state = forward(model, params, state, key, img)
    kl = kl_loss(z_mean, z_log_var)
    mse = mse_loss(img, reconstructed)
    loss = kl_weight * kl + mse
    return loss, (state, loss, kl, mse)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = VAE()
    params, state = init(model, init_key, r_train[:5], print_summary=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(lambda params, state, key, *x: forward(model, params, state, key, x[0])[0][0])
    train_metrics = ('loss', 'kl', 'mse')

    train_loop(
        'variational', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key
    )
