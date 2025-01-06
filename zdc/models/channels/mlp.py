from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.layers import LayerNormF32
from zdc.models import PARTICLE_TYPE, ParticleType
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step
from zdc.utils.train import train_loop, default_generate_fn


n_optimizer = optax.adam(3.7e-3, b1=0.53, b2=0.63)
p_optimizer = optax.adam(1.7e-3, b1=0.74, b2=0.22)


match PARTICLE_TYPE:
    case ParticleType.NEUTRON:
        optimizer = n_optimizer
    case ParticleType.PROTON:
        optimizer = p_optimizer
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

    def gen(self, x):
        return self(x)


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
    generate_fn = jax.jit(default_generate_fn(model))
    train_metrics = ('loss',)

    train_loop(
        'mlp_neutron', train_fn, 'channel', generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key
    )
