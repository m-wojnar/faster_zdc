"""
Note! The VQ-GAN export to ONNX is not supported yet. JAX (jax2tf.convert) does not allow exporting the model to TensorFlow.
"""

import jax
import numpy as np

from zdc.models import PARTICLE_TYPE
from zdc.models.quantization.gpt import GPTPrior
from zdc.models.quantization.vq_vae_gen import VQGAN, VQVAE
from zdc.utils.data import get_samples
from zdc.utils.metrics import Metrics
from zdc.utils.nn import forward, load_model


if __name__ == '__main__':
    if PARTICLE_TYPE == PARTICLE_TYPE.NEUTRON:
        prefix = 'neutron'
    elif PARTICLE_TYPE == PARTICLE_TYPE.PROTON:
        raise ValueError('VQ-GAN was not trained for the ZDC proton detector')

    ae_model = VQGAN()
    ae_params, ae_state = load_model(f'weights/{prefix}_ae.pkl.lz4')
    cond_model = VQVAE()
    cond_params, cond_state = load_model(f'weights/{prefix}_cond.pkl.lz4')
    gpt_model = GPTPrior()
    gpt_params, gpt_state = load_model(f'weights/{prefix}_gpt.pkl.lz4')
    batch_size = 7  # the number of sample particles

    c_empty = np.empty((batch_size, 2), dtype=np.int32)
    x_empty = np.empty((batch_size, 11 * 11), dtype=np.int32)
    cache = gpt_model.init({'params': jax.random.PRNGKey(0)}, c_empty, x_empty, False)['cache']
    gpt_state = gpt_state | {'cache': cache}

    @jax.jit
    def forward_fn(cond, seed):
        cond_key, gpt_key, ae_key = jax.random.split(jax.random.PRNGKey(seed), 3)
        context, _ = forward(cond_model, cond_params, cond_state, cond_key, cond, method='encode')
        latent, _ = forward(gpt_model, gpt_params, gpt_state, gpt_key, context, method='gen')
        generated, _ = forward(ae_model, ae_params, ae_state, ae_key, latent, method='gen')
        return generated

    responses, particles = get_samples()
    gen = forward_fn(particles, seed=100).astype(np.float32)
    Metrics(None, None, use_wandb=False).plot_responses(responses, gen, step=0)
