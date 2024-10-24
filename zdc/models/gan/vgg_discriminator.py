import jax.numpy as jnp
from flax import linen as nn
from flaxmodels.vgg import VGG16

from zdc.layers import Flatten


class FeatureExtractor(nn.Module):
    channels: int = None
    kernel_size_in: int = None
    kernel_size_out: int = None

    @nn.compact
    def __call__(self, x):
        if self.channels is not None:
            x = nn.Conv(
                features=self.channels,
                kernel_size=(self.kernel_size_in, self.kernel_size_in),
                strides=(self.kernel_size_in, self.kernel_size_in),
                padding='VALID',
                dtype=jnp.bfloat16,
                kernel_init=nn.initializers.zeros
            )(x)
            x = nn.relu(x)

        x = nn.Conv(
            features=1,
            kernel_size=(self.kernel_size_out, self.kernel_size_out),
            strides=(self.kernel_size_out, self.kernel_size_out),
            padding='VALID',
            dtype=jnp.bfloat16,
            kernel_init=nn.initializers.zeros
        )(x)
        x = Flatten()(x)

        return x


class Discriminator(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = vgg_preprocess(x)
        out = VGG16(output='activations', pretrained='imagenet', include_head=False, dtype=jnp.bfloat16)(x)
        fe1 = FeatureExtractor(channels=32, kernel_size_in=4, kernel_size_out=4)(out['relu1_2'])
        fe2 = FeatureExtractor(channels=64, kernel_size_in=4, kernel_size_out=2)(out['relu2_2'])
        fe3 = FeatureExtractor(channels=128, kernel_size_in=2, kernel_size_out=2)(out['relu3_3'])
        fe4 = FeatureExtractor(kernel_size_out=2)(out['relu4_3'])
        fe5 = FeatureExtractor(kernel_size_out=1)(out['relu5_3'])
        return (fe1 + fe2 + fe3 + fe4 + fe5).mean(axis=-1).astype(jnp.float32)


def vgg_preprocess(x, max_val=6.4):
    return 2 * (x / max_val) - 1
