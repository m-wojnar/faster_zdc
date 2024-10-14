import jax.numpy as jnp
from flax import linen as nn


class Patches(nn.Module):
    patch_size: int

    @nn.compact
    def __call__(self, x):
        b, h, w, c = x.shape
        x = x.reshape(b, h // self.patch_size, self.patch_size, w // self.patch_size, self.patch_size, c)
        x = x.transpose(0, 1, 3, 2, 4, 5)
        x = x.reshape(b, -1, *x.shape[3:])
        x = x.reshape(b, x.shape[1], -1)
        return x


class PatchEncoder(nn.Module):
    num_patches: int
    embedding_dim: int
    positional_encoding: bool = False

    @nn.compact
    def __call__(self, x, training=True):
        if self.positional_encoding:
            pos_embedding = nn.Embed(self.num_patches, self.embedding_dim)(jnp.arange(self.num_patches))
            x = nn.Dense(self.embedding_dim)(x)
            x = x + pos_embedding
        else:
            x = nn.Dense(self.embedding_dim)(x)

        return x
