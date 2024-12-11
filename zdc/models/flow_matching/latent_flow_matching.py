from functools import partial

import jax
import jax.numpy as jnp
import optax

from zdc.models import RESPONSE_SHAPE
from zdc.models.autoencoder.variational import VAE
from zdc.models.flow_matching.flow_matching import FMUnet, loss_fn
from zdc.utils.data import load, batches
from zdc.utils.nn import init, forward, load_model, gradient_step, opt_with_cosine_schedule
from zdc.utils.train import train_loop


optimizer = opt_with_cosine_schedule(partial(optax.adam, b1=0.9, b2=0.6), peak_value=1.3e-3)


class LatentFMUnet(FMUnet):
    channels: int = 8
    channel_multipliers: tuple = (1, 2, 4)
    n_resnet_blocks: int = 1
    n_heads: int = 2
    out_shape: tuple = (RESPONSE_SHAPE[0] // 4, RESPONSE_SHAPE[1] // 4, 4)


def encode_fn(x, batch_size, vae, vae_variables):
    encoded = [forward(vae, *vae_variables, jax.random.PRNGKey(0), *batch, method='encode')[0] for batch in batches(x, batch_size=batch_size)]
    return jnp.concatenate(encoded)


def latent_step_fn(params, carry, opt_state, optimizer, loss_fn):
    state, key, _, latent, cond = carry
    carry = (state, key, latent, cond)
    return gradient_step(params, carry, opt_state, optimizer, loss_fn)


def generate_fn(params, state, key, *x, latent_model, vae_model, vae_variables):
    z, _ = forward(latent_model, params, state, key, x[-1], 8, method='gen')
    x, _ = forward(vae_model, *vae_variables, key, z, method='gen')
    _, h, w, _ = x[0].shape
    return x[:, :h, :w]


if __name__ == '__main__':
    key = jax.random.PRNGKey(72)
    init_key, train_key = jax.random.split(key)
    batch_size = 256

    vae = VAE()
    vae_variables = load_model('../autoencoder/checkpoints/variational/epoch_50.pkl.lz4')
    vae_variables = (vae_variables[0][0], vae_variables[1][0])

    r_train, r_val, r_test, p_train, p_val, p_test = load()
    l_train, l_val, l_test = jax.tree.map(lambda x: encode_fn(x, batch_size, vae, vae_variables), (r_train, r_val, r_test))

    model = LatentFMUnet()
    params, state = init(model, init_key, l_train[:5], p_train[:5], jnp.empty(5), print_summary=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(latent_step_fn, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(partial(generate_fn, latent_model=model, vae_model=vae, vae_variables=vae_variables))
    train_metrics = ('loss', 'v_abs_mean', 'v_pred_abs_mean', 'gn')

    train_loop(
        'latent_flow_matching', train_fn, None, generate_fn, (r_train, l_train, p_train), (r_val, l_val, p_val), (r_test, l_test, p_test),
        train_metrics, None, params, state, opt_state, train_key
    )
