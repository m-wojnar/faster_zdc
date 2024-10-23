import jax
import jax.numpy as jnp
import optax
from lpips_j.lpips import LPIPS

from zdc.models import RESPONSE_SHAPE
from zdc.utils.data import vgg_preprocess
from zdc.utils.wasserstein import wasserstein_channels


def kl_loss(mean, log_var):
    return -0.5 * (1. + log_var - jnp.square(mean) - jnp.exp(log_var)).mean()


def mse_loss(x, y):
    return jnp.square(x - y).mean()


def mae_loss(x, y):
    return jnp.abs(x - y).mean()


def wasserstein_loss(ch_true, ch_pred):
    return wasserstein_channels(ch_true, ch_pred).mean()


def xentropy_loss(x, y):
    return optax.sigmoid_binary_cross_entropy(x, y).mean()


def perceptual_loss():
    lpips = LPIPS()
    x_sample = jnp.zeros((1, *RESPONSE_SHAPE))
    params = lpips.init(jax.random.PRNGKey(0), x_sample, x_sample)

    def apply(x, y):
        x, y = vgg_preprocess(x), vgg_preprocess(y)
        return lpips.apply(params, x, y).mean()

    return apply
