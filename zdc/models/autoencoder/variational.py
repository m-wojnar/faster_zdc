from functools import partial

import jax
import optax
from flax import linen as nn

from zdc.architectures.conv import Encoder, Decoder
from zdc.layers import Conv, Sampling
from zdc.models.gan.vgg_discriminator import Discriminator
from zdc.utils.data import load
from zdc.utils.grad import grad_norm
from zdc.utils.losses import kl_loss, mse_loss, perceptual_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


gen_optimizer = opt_with_cosine_schedule(partial(optax.adam, b1=0.58, b2=0.75), peak_value=2.3e-3)
disc_optimizer = opt_with_cosine_schedule(partial(optax.adam, b1=0.44, b2=0.77), peak_value=1.4e-6)


class VAE(nn.Module):
    channels: int = 4
    z_channels: int = 4
    channel_multipliers: tuple = (2, 3, 4)
    n_resnet_blocks: int = 2
    n_heads: int = 2

    def setup(self):
        self.encoder = Encoder(self.channels, self.channel_multipliers, self.n_resnet_blocks, self.n_heads)
        self.z_mean_conv = Conv(self.z_channels, kernel_size=1)
        self.z_log_var_conv = Conv(self.z_channels, kernel_size=1)
        self.sampling = Sampling()
        self.decoder = Decoder(self.channels, self.channel_multipliers, self.n_resnet_blocks, self.n_heads)

    def __call__(self, img):
        z = self.encoder(img)
        z_mean = self.z_mean_conv(z)
        z_log_var = self.z_log_var_conv(z)
        z = self.sampling(z_mean, z_log_var)
        return self.decoder(z), z_mean, z_log_var

    def encode(self, img):
        z = self.encoder(img)
        z_mean = self.z_mean_conv(z)
        return z_mean

    def gen(self, z):
        return self.decoder(z)


def disc_loss_fn(disc_params, disc_state, gen_params, gen_state, key, *x, gen_model, disc_model):
    gen_key, disc_real_key, disc_fake_key = jax.random.split(key, 3)
    img, rand_img = x

    (generated, *_), _ = forward(gen_model, gen_params, gen_state, gen_key, img)
    real_output, disc_state = forward(disc_model, disc_params, disc_state, disc_real_key, rand_img)
    fake_output, disc_state = forward(disc_model, disc_params, disc_state, disc_fake_key, generated)

    real = nn.relu(1 - real_output).mean()
    fake = nn.relu(1 + fake_output).mean()
    real_logits = real_output.mean()
    fake_logits = fake_output.mean()

    loss = real + fake
    return loss, (disc_state, loss, real_logits, fake_logits)


def gen_loss_fn(gen_params, gen_state, disc_params, disc_state, key, *x, gen_model, disc_model, lpips_fn):
    gen_key, disc_key = jax.random.split(key)
    img, *_ = x

    (generated, z_mean, z_log_var), gen_state = forward(gen_model, gen_params, gen_state, gen_key, img)

    def vae_loss(generated, img, z_mean, z_log_var):
        generated = grad_norm(generated, 0.1)
        l2 = mse_loss(img, generated)
        kl = kl_loss(z_mean, z_log_var)
        return l2 + kl

    def perc_loss(generated, img):
        generated = grad_norm(generated, 1.0)
        return lpips_fn(img, generated)

    def adv_loss(generated):
        generated = grad_norm(generated, 1.0)
        fake_output, _ = forward(disc_model, disc_params, disc_state, disc_key, generated)
        return -fake_output.mean()

    vae = vae_loss(generated, img, z_mean, z_log_var)
    perc = perc_loss(generated, img)
    adv = adv_loss(generated)
    loss = vae + perc + adv

    return loss, (gen_state, loss, vae, adv, perc)


def step_fn(params, carry, opt_state, disc_optimizer, gen_optimizer, disc_loss_fn, gen_loss_fn):
    gen_params, disc_params = params
    (gen_state, disc_state), key, img, *_ = carry
    gen_opt_state, disc_opt_state = opt_state
    gen_key, disc_key, data_key = jax.random.split(key, 3)
    rand_img = jax.random.permutation(data_key, img)

    disc_params_new, disc_opt_state, disc_grads, (disc_state_new, *disc_losses) = gradient_step(
        disc_params, (disc_state, gen_params, gen_state, disc_key, img, rand_img), disc_opt_state, disc_optimizer, disc_loss_fn)
    gen_params_new, gen_opt_state, gen_grads, (gen_state_new, *gen_losses) = gradient_step(
        gen_params, (gen_state, disc_params, disc_state, gen_key, img, rand_img), gen_opt_state, gen_optimizer, gen_loss_fn)

    disc_gn = optax.tree_utils.tree_l2_norm(disc_grads)
    gen_gn = optax.tree_utils.tree_l2_norm(gen_grads)

    return (gen_params_new, disc_params_new), (gen_opt_state, disc_opt_state), None, ((gen_state_new, disc_state_new), *disc_losses, *gen_losses, disc_gn, gen_gn)


if __name__ == '__main__':
    key = jax.random.PRNGKey(72)
    gen_init_key, disc_init_key, train_key = jax.random.split(key, 3)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    gen_model = VAE()
    gen_params, gen_state = init(gen_model, gen_init_key, r_train[:5], print_summary=True)
    gen_opt_state = gen_optimizer.init(gen_params)

    disc_model = Discriminator()
    disc_params, disc_state = init(disc_model, disc_init_key, r_train[:5], print_summary=True)
    disc_opt_state = disc_optimizer.init(disc_params)

    train_fn = jax.jit(partial(
        step_fn,
        disc_optimizer=disc_optimizer,
        gen_optimizer=gen_optimizer,
        disc_loss_fn=partial(disc_loss_fn, gen_model=gen_model, disc_model=disc_model),
        gen_loss_fn=partial(gen_loss_fn, gen_model=gen_model, disc_model=disc_model, lpips_fn=perceptual_loss())
    ))
    generate_fn = jax.jit(lambda params, state, key, *x: forward(gen_model, params[0], state[0], key, x[0])[0][0])
    train_metrics = ('disc_loss', 'real_logits', 'fake_logits', 'gen_loss', 'vae_loss', 'adv_loss', 'perc_loss', 'disc_gn', 'gen_gn')

    train_loop(
        'variational', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, (gen_params, disc_params), (gen_state, disc_state), (gen_opt_state, disc_opt_state), train_key
    )
