from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import Flatten, Reshape, VectorQuantizer
from zdc.models import PARTICLE_SHAPE
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


optimizer = opt_with_cosine_schedule(partial(optax.adamw, b1=0.71, b2=0.88, weight_decay=0.03), 1.7e-3)


class Encoder(nn.Module):
    hidden_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, cond):
        x = nn.Dense(self.hidden_dim, dtype=jnp.bfloat16)(cond)
        x = nn.gelu(x)
        x = nn.Dense(self.latent_dim * self.hidden_dim, dtype=jnp.bfloat16)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.latent_dim * self.hidden_dim, dtype=jnp.bfloat16)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.latent_dim * self.hidden_dim, dtype=jnp.bfloat16)(x)
        x = nn.gelu(x)
        x = Reshape((self.latent_dim, self.hidden_dim))(x)
        return x


class Decoder(nn.Module):
    hidden_dim: int
    latent_dim: int

    @nn.compact
    def __call__(self, z):
        x = Flatten()(z)
        x = nn.Dense(self.latent_dim * self.hidden_dim, dtype=jnp.bfloat16)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.latent_dim * self.hidden_dim, dtype=jnp.bfloat16)(x)
        x = nn.gelu(x)
        x = nn.Dense(self.hidden_dim, dtype=jnp.bfloat16)(x)
        x = nn.gelu(x)
        x = nn.Dense(*PARTICLE_SHAPE, dtype=jnp.bfloat16)(x)
        return x


class VQVAE(nn.Module):
    hidden_dim: int = 64
    latent_dim: int = 2
    proj_dim: int = 4
    num_embeddings: int = 512
    normalize: bool = True

    def setup(self) -> None:
        self.encoder = Encoder(self.hidden_dim, self.latent_dim)
        self.quantizer = VectorQuantizer(self.num_embeddings, self.hidden_dim, self.proj_dim, self.normalize)
        self.decoder = Decoder(self.hidden_dim, self.latent_dim)

    def __call__(self, cond):
        encoded = self.encoder(cond)
        discrete, quantized = self.quantizer(encoded)
        encoded = VectorQuantizer.l2_normalize(encoded) if self.normalize else encoded
        quantized_sg = encoded + jax.lax.stop_gradient(quantized - encoded)
        return self.decoder(quantized_sg), encoded, discrete, quantized


def loss_fn(params, state, key, img, cond, model):
    (reconstructed, encoded, discrete, quantized), state = forward(model, params, state, key, cond)

    e_loss = mse_loss(jax.lax.stop_gradient(quantized), encoded)
    q_loss = mse_loss(quantized, jax.lax.stop_gradient(encoded))
    l2 = mse_loss(cond, reconstructed)
    loss = l2 + 0.25 * e_loss + q_loss

    avg_prob = jnp.mean(discrete, axis=0)
    perplexity = jnp.exp(-jnp.sum(avg_prob * jnp.log(avg_prob + 1e-10)))

    return loss, (state, loss, l2, e_loss, q_loss, perplexity)


def eval_fn(generated, *dataset):
    _, cond = dataset
    return (mse_loss(cond, generated),)


if __name__ == '__main__':
    key = jax.random.PRNGKey(72)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = VQVAE()
    params, state = init(model, init_key, p_train[:5], print_summary=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(lambda params, state, key, *x: forward(model, params, state, key, x[1])[0][0])
    train_metrics = ('loss', 'l2_loss', 'e_loss', 'q_loss', 'perplexity')
    eval_metrics = ('mse',)

    train_loop(
        'vq_vae_cond', train_fn, eval_fn, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, eval_metrics, params, state, opt_state, train_key, n_rep=1
    )
