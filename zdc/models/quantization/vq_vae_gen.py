from functools import partial

import jax
import jax.numpy as jnp

from zdc.models.quantization.vq_gan import VQVAE as VQGAN
from zdc.models.quantization.vq_vae_cond import VQVAE
from zdc.models.quantization.gpt import GPTPrior, tokenize_fn
from zdc.utils.data import load, batches, get_samples
from zdc.utils.metrics import Metrics
from zdc.utils.nn import forward, load_model
from zdc.utils.train import default_eval_fn


def generate_fn(key, c, vq_gan, vq_gan_variables, vq_prior, vq_prior_variables):
    prior_key, decoder_key = jax.random.split(key)
    x_empty = jnp.empty((c.shape[0], 11 * 11), dtype=jnp.int32)
    cache = vq_prior.init({'params': jax.random.PRNGKey(0)}, c, x_empty, False)['cache']
    vq_prior_variables = (vq_prior_variables[0], vq_prior_variables[1] | {'cache': cache})

    generated, _ = forward(vq_prior, *vq_prior_variables, prior_key, c, method='gen')
    generated, _ = forward(vq_gan, *vq_gan_variables, decoder_key, generated, method='gen')

    return generated


if __name__ == '__main__':
    key = jax.random.PRNGKey(72)
    tokenize_key, test_key, plot_key = jax.random.split(key, 3)
    batch_size = 256
    n_rep = 5

    vq_gan = VQGAN()
    vq_vae_cond = VQVAE()
    vq_prior = GPTPrior()
    vq_gan_variables = load_model('checkpoints/vq_gan/epoch_50.pkl.lz4')
    vq_gan_variables = (vq_gan_variables[0][0], vq_gan_variables[1][0])
    vq_vae_cond_variables = load_model('checkpoints/vq_vae_cond/epoch_50.pkl.lz4')
    vq_prior_variables = load_model('checkpoints/gpt/epoch_50.pkl.lz4')

    vq_vae_cond_fn = jax.jit(lambda *args: forward(vq_vae_cond, *vq_vae_cond_variables, *args)[0][2])
    gen_fn = jax.jit(partial(generate_fn, vq_gan=vq_gan, vq_gan_variables=vq_gan_variables, vq_prior=vq_prior, vq_prior_variables=vq_prior_variables))
    tok_fn = partial(tokenize_fn, batch_size=batch_size, model_fn=vq_vae_cond_fn)

    _, _, r_test, _, _, p_test = load()
    r_sample, p_sample = get_samples()
    c_test, c_sample = jax.tree.map(tok_fn, tuple(jax.random.split(tokenize_key)), (p_test, p_sample))

    metrics = Metrics(job_type='train', name='vq_vae_gen')
    eval_metrics = ('rmse', 'mae', 'wasserstein')
    generated, original = [], []

    for r_batch, c_batch in batches(r_test, c_test, batch_size=batch_size):
        for _ in range(n_rep):
            test_key, subkey = jax.random.split(test_key)
            generated.append(gen_fn(subkey, c_batch))
            original.append(r_batch)

    generated, original = jnp.concatenate(generated), jnp.concatenate(original)
    metrics.add(dict(zip(eval_metrics, default_eval_fn(generated, original))), 'test')
    metrics.plot_responses(r_sample, gen_fn(plot_key, c_sample).astype(jnp.float32), 0)
    metrics.log(0)
