from functools import partial
from typing import List

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn
from flaxmodels.vgg import VGG16

from zdc.layers import Sampling, UpSample, Flatten
from zdc.utils.data import load, vgg_preprocess
from zdc.utils.losses import kl_loss, mse_loss, perceptual_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule, get_layers
from zdc.utils.train import train_loop


gen_optimizer = opt_with_cosine_schedule(
    optimizer=partial(optax.adam),
    peak_value=1e-5,
    epochs=50,
    batch_size=256
)
disc_optimizer = optax.adam(2e-4)


class BinaryClassifier(nn.Module):
    channels: int = None
    kernel_size_in: int = None
    kernel_size_out: int = None
    
    @nn.compact
    def __call__(self, x):
        if self.channels is not None:
            x = nn.Conv(
                features=self.channels,
                kernel_size=(self.kernel_size_in, self.kernel_size_in),
                strides=(self.kernel_size_in, self.kernel_size_in),
                padding='VALID',
                dtype=jnp.bfloat16,
                kernel_init=nn.initializers.zeros
            )(x)
            x = nn.relu(x)

        x = nn.Conv(
            features=1,
            kernel_size=(self.kernel_size_out, self.kernel_size_out),
            strides=(self.kernel_size_out, self.kernel_size_out),
            padding='VALID',
            dtype=jnp.bfloat16,
            kernel_init=nn.initializers.zeros
        )(x)
        x = Flatten()(x)

        return x


class Discriminator(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = vgg_preprocess(x)
        out = VGG16(output='activations', pretrained='imagenet', include_head=False, dtype=jnp.bfloat16)(x)
        bc1 = BinaryClassifier(channels=32, kernel_size_in=4, kernel_size_out=4)(out['relu1_2'])
        bc2 = BinaryClassifier(channels=64, kernel_size_in=4, kernel_size_out=2)(out['relu2_2'])
        bc3 = BinaryClassifier(channels=128, kernel_size_in=2, kernel_size_out=2)(out['relu3_3'])
        bc4 = BinaryClassifier(kernel_size_out=2)(out['relu4_3'])
        bc5 = BinaryClassifier(kernel_size_out=1)(out['relu5_3'])
        return (bc1 + bc2 + bc3 + bc4 + bc5).mean(axis=-1)


class VAE(nn.Module):
    channels: int = 4
    z_channels: int = 4
    channel_multipliers: tuple = (2, 3, 4)
    n_heads: int = 2
    n_resnet_blocks: int = 2

    @nn.compact
    def __call__(self, img):
        z, z_mean, z_log_var = self.encode(img)
        return self.decode(z), z_mean, z_log_var

    def encode(self, img):
        z = Encoder(self.channels, self.channel_multipliers, self.n_heads, self.n_resnet_blocks)(img)
        z_mean = Conv(self.z_channels, kernel_size=1)(z)
        z_log_var = Conv(self.z_channels, kernel_size=1)(z)
        return Sampling()(z_mean, z_log_var), z_mean, z_log_var

    def decode(self, z):
        return Decoder(self.channels, self.channel_multipliers, self.n_heads, self.n_resnet_blocks)(z)

    def gen(self, z):
        return self.decode(z)


class Encoder(nn.Module):
    channels: int
    channel_multipliers: List[int]
    n_heads: int
    n_resnet_blocks: int

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
    n_heads: int = 4

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


class Model(nn.Module):
    def setup(self):
        self.discriminator = Discriminator()
        self.generator = VAE()

    def __call__(self, img, rand_img):
        reconstructed, z_mean, z_log_var = self.generator(img)
        real_output = self.discriminator(rand_img).astype(jnp.float32)
        fake_output = self.discriminator(reconstructed).astype(jnp.float32)
        return reconstructed, z_mean, z_log_var, real_output, fake_output

    def gen(self, img):
        return self.generator(img)[0]


def disc_loss_fn(disc_params, gen_params, state, forward_key, *x, model):
    (*_, real_output, fake_output), state = forward(model, disc_params | gen_params, state, forward_key, *x)

    real = nn.relu(1 - real_output).mean()
    fake = nn.relu(1 + fake_output).mean()
    disc_real_acc = (real_output > 0).mean()
    disc_fake_acc = (fake_output < 0).mean()

    loss = real + fake
    return loss, (state, loss, disc_real_acc, disc_fake_acc)


def gen_loss_fn(gen_params, disc_params, state, forward_key, *x, model, loss_weights, lpips_fn):
    (generated, z_mean, z_log_var, _, fake_output), state = forward(model, gen_params | disc_params, state, forward_key, *x)
    img, *_ = x

    l2 = mse_loss(img, generated)
    kl = kl_loss(z_mean, z_log_var)
    perc = lpips_fn(img, generated)
    adv = -fake_output.mean()
    gen_acc = (fake_output > 0).mean()

    l2_weight, kl_weight, adv_weight, perc_weight = loss_weights
    loss = l2_weight * l2 + kl_weight * kl + adv_weight * adv + perc_weight * perc
    return loss, (state, loss, l2, kl, adv, perc, gen_acc)


def step_fn(params, carry, opt_state, disc_optimizer, gen_optimizer, disc_loss_fn, gen_loss_fn):
    (state, key, img, _), (disc_opt_state, gen_opt_state) = carry, opt_state
    forward_key, data_key = jax.random.split(key)

    disc_params, gen_params = get_layers(params, 'discriminator'), get_layers(params, 'generator')
    permutation = jax.random.permutation(data_key, img.shape[0])
    rand_img = img[permutation]

    disc_params_new, disc_opt_state, disc_grads, (_, *disc_losses) = gradient_step(
        disc_params, (gen_params, state, forward_key, img, rand_img), disc_opt_state, disc_optimizer, disc_loss_fn)
    gen_params_new, gen_opt_state, gen_grads, (state, *gen_losses) = gradient_step(
        gen_params, (disc_params, state, forward_key, img, rand_img), gen_opt_state, gen_optimizer, gen_loss_fn)

    disc_gn = optax.tree_utils.tree_l2_norm(disc_grads)
    gen_gn = optax.tree_utils.tree_l2_norm(gen_grads)

    return disc_params_new | gen_params_new, (disc_opt_state, gen_opt_state), (state, *disc_losses, *gen_losses, disc_gn, gen_gn)


if __name__ == '__main__':
    key = jax.random.PRNGKey(42)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = Model()
    params, state = init(model, init_key, r_train[:5], r_train[:5], print_summary=True)
    disc_opt_state = disc_optimizer.init(get_layers(params, 'discriminator'))
    gen_opt_state = gen_optimizer.init(get_layers(params, 'generator'))

    train_fn = jax.jit(partial(
        step_fn,
        disc_optimizer=disc_optimizer,
        gen_optimizer=gen_optimizer,
        disc_loss_fn=partial(disc_loss_fn, model=model),
        gen_loss_fn=partial(gen_loss_fn, model=model, loss_weights=(1.0, 0.7, 0.2, 1.0), lpips_fn=perceptual_loss())
    ))
    generate_fn = jax.jit(lambda params, state, key, *x: forward(model, params, state, key, x[0], method='gen')[0])
    train_metrics = ('disc_loss', 'disc_real_acc', 'disc_fake_acc', 'gen_loss', 'l2_loss', 'kl_loss', 'adv_loss', 'perc_loss', 'gen_acc', 'disc_gn', 'gen_gn')

    train_loop(
        'variational_gan', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, (disc_opt_state, gen_opt_state), train_key, epochs=50
    )
