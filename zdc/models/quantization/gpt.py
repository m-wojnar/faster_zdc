from functools import partial

import jax
import jax.numpy as jnp
import optax

from zdc.architectures.transformer import GPT
from zdc.models.quantization.vq_gan import VQVAE as VQGAN
from zdc.models.quantization.vq_vae_cond import VQVAE as VQVAE
from zdc.utils.data import load, batches
from zdc.utils.losses import xentropy_loss
from zdc.utils.nn import init, forward, gradient_step, opt_with_cosine_schedule, load_model
from zdc.utils.train import train_loop


optimizer = opt_with_cosine_schedule(partial(optax.adamw, b1=0.80, b2=0.65, eps=1.7e-9, weight_decay=0.33), peak_value=3.6e-3)


def loss_fn(params, state, key, c, x, y, model):
    logits, state = forward(model, params, state, key, c, x)
    loss = xentropy_loss(logits, y)
    perplexity = jnp.exp(loss)
    return loss, (state, loss, perplexity)


def eval_fn(generated, *dataset):
    c, _, y = dataset
    generated = jax.nn.one_hot(generated, 512).astype(jnp.float32)
    loss = xentropy_loss(generated, y[:, c.shape[1] - 1:])
    perplexity = jnp.exp(loss)
    return loss, perplexity


def tokenize_fn(key, x, batch_size, model_fn):
    tokenized = []

    for batch in batches(x, batch_size=batch_size):
        key, subkey = jax.random.split(key)
        _, discrete = jnp.where(model_fn(subkey, *batch))
        tokenized.append(discrete.reshape(batch[0].shape[0], -1))

    return jnp.concatenate(tokenized)


if __name__ == '__main__':
    key = jax.random.PRNGKey(72)
    init_key, r_key, p_key, train_key = jax.random.split(key, 4)
    batch_size = 256

    vq_gan = VQGAN()
    vq_vae_cond = VQVAE()
    vq_gan_variables = load_model('checkpoints/vq_gan/epoch_50.pkl.lz4')
    vq_gan_variables = (vq_gan_variables[0][0], vq_gan_variables[1][0])
    vq_vae_cond_variables = load_model('checkpoints/vq_vae_cond/epoch_50.pkl.lz4')

    vq_gan_fn = jax.jit(lambda *args: forward(vq_gan, *vq_gan_variables, *args)[0][2])
    vq_vae_cond_fn = jax.jit(lambda *args: forward(vq_vae_cond, *vq_vae_cond_variables, *args)[0][2])

    r_train, r_val, r_test, p_train, p_val, p_test = load()
    r_train, r_val, r_test = jax.tree.map(lambda x: tokenize_fn(r_key, x, batch_size, vq_gan_fn), (r_train, r_val, r_test))
    c_train, c_val, c_test = jax.tree.map(lambda x: tokenize_fn(p_key, x, batch_size, vq_vae_cond_fn), (p_train, p_val, p_test))
    x_train, x_val, x_test = jax.tree.map(lambda x: x[:, :-1], (r_train, r_val, r_test))
    y_train, y_val, y_test = jax.tree.map(lambda c, x: jnp.concatenate((c, x), axis=1)[:, 1:], (c_train, c_val, c_test), (r_train, r_val, r_test))

    c_val, c_test, x_val, x_test, y_val, y_test = jax.tree.map(lambda x: x[:-(x.shape[0] % batch_size)], (c_val, c_test, x_val, x_test, y_val, y_test))

    model = GPT(vocab_size=512, embed_dim=64, seq_len=121, max_len=122, hidden_dim=128, num_heads=2, num_layers=2, drop_rate=0.1)
    params, state = init(model, init_key, c_train[:5], x_train[:5], print_summary=True)
    cache = model.init({'params': jax.random.PRNGKey(0)}, c_train[:batch_size], x_train[:batch_size], False)['cache']
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    gen_fn = jax.jit(lambda params, state, key, *x: forward(model, params, state | {'cache': cache}, key, x[0], method='gen')[0])
    metrics = ('loss', 'perplexity')

    train_loop(
        'gpt', train_fn, eval_fn, gen_fn, (c_train, x_train, y_train), (c_val, x_val, y_val), (c_test, x_test, y_test),
        metrics, metrics, params, state, opt_state, train_key, n_rep=1
    )
