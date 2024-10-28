from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.architectures.conv import Encoder, Decoder
from zdc.layers import Conv, VectorQuantizerEMA
from zdc.models.autoencoder.variational import step_fn, disc_loss_fn
from zdc.models.gan.vgg_discriminator import Discriminator
from zdc.utils.data import load
from zdc.utils.grad import grad_norm
from zdc.utils.losses import mse_loss, perceptual_loss
from zdc.utils.nn import init, forward, opt_with_cosine_schedule
from zdc.utils.train import train_loop


gen_optimizer = opt_with_cosine_schedule(partial(optax.adam, b1=0.55, b2=0.62), peak_value=8.8e-4)
disc_optimizer = opt_with_cosine_schedule(partial(optax.adam, b1=0.45, b2=0.88), peak_value=1.1e-6)


class VQVAE(nn.Module):
    channels: int = 4
    emb_channels: int = 8
    channel_multipliers: tuple = (2, 3, 4)
    n_resnet_blocks: int = 2
    n_heads: int = 2
    num_embeddings: int = 512
    normalize: bool = True

    def setup(self) -> None:
        self.encoder = Encoder(self.channels, self.channel_multipliers, self.n_resnet_blocks, self.n_heads)
        self.emb = Conv(self.emb_channels, kernel_size=1)
        self.quantizer = VectorQuantizerEMA(self.num_embeddings, self.emb_channels, normalize=self.normalize)
        self.decoder = Decoder(self.channels, self.channel_multipliers, self.n_resnet_blocks, self.n_heads)

    def __call__(self, img):
        z = self.encoder(img)
        encoded = self.emb(z)
        discrete, quantized = self.quantizer(encoded)
        encoded = VectorQuantizerEMA.l2_normalize(encoded) if self.normalize else encoded
        quantized_sg = encoded + jax.lax.stop_gradient(quantized - encoded)
        return self.decoder(quantized_sg), encoded, discrete, quantized

    def gen(self, discrete):
        discrete = nn.one_hot(discrete, self.num_embeddings)
        quantized = self.quantizer.quantize(discrete)
        quantized = quantized.reshape(-1, 11, 11, self.emb_channels)
        quantized = VectorQuantizerEMA.l2_normalize(quantized) if self.normalize else quantized
        return self.decoder(quantized)


def gen_loss_fn(gen_params, gen_state, disc_params, disc_state, key, *x, gen_model, disc_model, lpips_fn):
    gen_key, disc_key = jax.random.split(key)
    img, *_ = x

    (generated, encoded, discrete, quantized), gen_state = forward(gen_model, gen_params, gen_state, gen_key, img)

    def vq_loss(generated, img, encoded, quantized):
        generated = grad_norm(generated, 0.1)
        l2 = mse_loss(img, generated)
        vq = 0.25 * mse_loss(jax.lax.stop_gradient(quantized), encoded)
        return l2 + vq

    def perc_loss(generated, img):
        generated = grad_norm(generated, 1.0)
        return lpips_fn(img, generated)

    def adv_loss(generated):
        generated = grad_norm(generated, 1.0)
        fake_output, _ = forward(disc_model, disc_params, disc_state, disc_key, generated)
        return -fake_output.mean()

    vq = vq_loss(generated, img, encoded, quantized)
    perc = perc_loss(generated, img)
    adv = adv_loss(generated)
    loss = vq + perc + adv

    avg_prob = jnp.mean(discrete, axis=0)
    perplexity = jnp.exp(-jnp.sum(avg_prob * jnp.log(avg_prob + 1e-10)))

    return loss, (gen_state, loss, vq, adv, perc, perplexity)


if __name__ == '__main__':
    key = jax.random.PRNGKey(72)
    gen_init_key, disc_init_key, train_key = jax.random.split(key, 3)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    gen_model = VQVAE()
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
    train_metrics = ('disc_loss', 'real_logits', 'fake_logits', 'gen_loss', 'vq_loss', 'adv_loss', 'perc_loss', 'perplexity', 'disc_gn', 'gen_gn')

    train_loop(
        'vq_gan', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, (gen_params, disc_params), (gen_state, disc_state), (gen_opt_state, disc_opt_state), train_key
    )
