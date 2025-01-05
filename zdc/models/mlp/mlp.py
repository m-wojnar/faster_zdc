from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import LayerNormF32
from zdc.models import PARTICLE_TYPE, ParticleType
from zdc.utils.data import load
from zdc.utils.losses import mse_loss, wasserstein_loss
from zdc.utils.nn import init, forward, gradient_step
from zdc.utils.train import train_loop


n_optimizer = optax.adam(4.4e-3, b1=0.25, b2=0.98)


match PARTICLE_TYPE:
    case ParticleType.NEUTRON:
        optimizer = n_optimizer
    case ParticleType.PROTON:
        pass
    case _:
        raise ValueError('Invalid particle type')


class MLP(nn.Module):
    dim: int = 64
    n_layers: int = 6
    drop_rate: float = 0.2
    
    @nn.compact
    def __call__(self, c, training=True):
        dense = partial(nn.Dense, dtype=jnp.bfloat16)
        dropout = partial(nn.Dropout, rate=self.drop_rate)
        
        x = dense(self.dim)(c)
        x = nn.gelu(x)
        x = dropout()(x, deterministic=not training)

        for i in range(self.n_layers - 3):
            x_res = x
            x = LayerNormF32()(x)
            x = dense(self.dim)(x)
            x = nn.gelu(x)
            x = dropout()(x, deterministic=not training)
            x = x + x_res

        x = LayerNormF32()(x)
        x = dense(self.dim)(x)
        x = nn.gelu(x)

        x = LayerNormF32()(x)
        x = dense(5)(x)
        x = nn.relu(x)
        return x


def eval_fn(generated, *dataset):
    ch_true, *_ = dataset
    ch_true, ch_pred = jnp.exp(ch_true) - 1, jnp.exp(generated) - 1
    return (wasserstein_loss(ch_true, ch_pred),)


def loss_fn(params, state, key, x, c, model):
    pred, state = forward(model, params, state, key, c)
    loss = mse_loss(pred, x)
    return loss, (state, loss)


if __name__ == '__main__':
    key = jax.random.PRNGKey(72)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load(add_ch_dim=False)

    model = MLP()
    params, state = init(model, init_key, p_train[:5], print_summary=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    eval_fn = jax.jit(eval_fn)
    generate_fn = jax.jit(lambda params, state, key, *x: forward(model, params, state, key, x[1])[0])
    train_metrics = ('loss',)
    eval_metrics = ('wasserstein',)

    train_loop(
        'mlp', train_fn, eval_fn, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, eval_metrics, params, state, opt_state, train_key
    )
