from functools import partial

import jax
import jax.numpy as jnp
import optax
from flax import linen as nn

from zdc.architectures.unet import UNet
from zdc.models import RESPONSE_SHAPE
from zdc.utils.data import load
from zdc.utils.losses import mse_loss
from zdc.utils.nn import init, forward, gradient_step
from zdc.utils.train import train_loop


optimizer = optax.adam(4.4e-4, b1=0.7, b2=0.88)


class FMUnet(UNet):
    channels: int = 4
    channel_multipliers: tuple = (2, 3, 4)
    n_resnet_blocks: int = 2
    n_heads: int = 2
    out_shape: tuple = RESPONSE_SHAPE

    def gen(self, cond, n_steps=11):
        def scan_fn(unet, x, t):
            t = jnp.full(cond.shape[0], t)
            v = unet(x, cond, t)
            x = x + v / n_steps
            return x, None

        z = jax.random.normal(self.make_rng('zdc'), (cond.shape[0], *self.out_shape))
        scan = nn.scan(scan_fn, variable_broadcast='params')
        x, _ = scan(self, z, jnp.linspace(0.0, 1.0, n_steps, endpoint=False))
        return x

    def gen_zdc(self, cond, n_steps=11):
        x = self.gen(cond, n_steps)
        x = nn.relu(x)
        return x


def loss_fn(params, state, key, img, cond, model, schedule='uniform'):
    z_key, t_key, model_key = jax.random.split(key, 3)

    if schedule == 'log-normal':
        t = jax.random.normal(t_key, (img.shape[0],))
        t = 1 / (1 + jnp.exp(-t))
    elif schedule == 'uniform':
        t = jax.random.uniform(t_key, (img.shape[0],), minval=0.0, maxval=0.99)
    else:
        raise ValueError(f'Invalid schedule: {schedule}')

    z = jax.random.normal(z_key, img.shape)
    t_b = t[..., None, None, None]

    x_t = (1 - t_b) * z + t_b * img
    v_t = img - z

    v_pred, state = forward(model, params, state, model_key, x_t, cond, t)

    v_abs_mean = jnp.abs(v_t).mean()
    v_pred_abs_mean = jnp.abs(v_pred).mean()

    loss = mse_loss(v_pred, v_t)
    return loss, (state, loss, v_abs_mean, v_pred_abs_mean)


if __name__ == '__main__':
    key = jax.random.PRNGKey(72)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load()

    model = FMUnet()
    params, state = init(model, init_key, r_train[:5], p_train[:5], jnp.empty(5), print_summary=True)
    opt_state = optimizer.init(params)

    train_fn = jax.jit(partial(gradient_step, optimizer=optimizer, loss_fn=partial(loss_fn, model=model)))
    generate_fn = jax.jit(lambda params, state, key, *x: forward(model, params, state, key, x[1], method='gen_zdc')[0])
    train_metrics = ('loss', 'v_abs_mean', 'v_pred_abs_mean')

    train_loop(
        'flow_matching', train_fn, None, generate_fn, (r_train, p_train), (r_val, p_val), (r_test, p_test),
        train_metrics, None, params, state, opt_state, train_key
    )
