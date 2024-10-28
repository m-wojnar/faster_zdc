import jax
import jax.numpy as jnp
from flax import linen as nn

from zdc.layers import TransformerBlock


class Transformer(nn.Module):
    vocab_size: int
    embed_dim: int
    seq_len: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    drop_rate: float
    decode: bool

    def setup(self) -> None:
        self.token_emb = nn.Embed(self.vocab_size, self.embed_dim)
        self.pos_emb = nn.Embed(self.seq_len, self.embed_dim)
        self.t_blocks = [
            TransformerBlock(self.num_heads, self.hidden_dim, self.drop_rate, self.decode)
            for _ in range(self.num_layers)
        ]
        self.out_ln = nn.LayerNorm(use_bias=False, dtype=jnp.bfloat16)

    def __call__(self, x, pos, mask, training=True):
        x = self.token_emb(x)
        x = x + self.pos_emb(pos)

        for block in self.t_blocks:
            x = block(x, mask, training=training)

        x = self.out_ln(x)
        x = self.token_emb.attend(x.astype(jnp.float32))

        return x


class GPT(nn.Module):
    vocab_size: int
    embed_dim: int
    seq_len: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    drop_rate: float

    @nn.compact
    def __call__(self, x, training=True):
        if not training:
            is_initialized = self.has_variable('cache', 'cache_index')
            cache_index = self.variable('cache', 'cache_index', lambda: jnp.array(0, dtype=jnp.int32))

            if is_initialized:
                i = cache_index.value
                cache_index.value = i + 1
            else:
                i = jnp.array(0, dtype=jnp.int32)

            pos = i
            mask = None
        else:
            pos = jnp.arange(x.shape[1])
            mask = nn.make_causal_mask(x, dtype=jnp.bfloat16)

        return Transformer(
            self.vocab_size, self.embed_dim, self.seq_len, self.hidden_dim, self.num_heads, self.num_layers, self.drop_rate, decode=not training
        )(x, pos, mask, training=training)

    def gen(self, cond):
        def scan_cond_fn(gpt, _, token):
            _ = gpt(token, training=False)
            return token, token

        def scan_fn(gpt, carry):
            prev_token, key = carry
            key, cat_key = jax.random.split(key)
            logits = gpt(prev_token, training=False)
            next_token = jax.random.categorical(cat_key, logits)
            return (next_token, key), next_token

        scan_cond = nn.scan(scan_cond_fn, variable_broadcast='params', variable_carry='cache', out_axes=1)
        scan_cond(self, None, cond)

        scan = nn.scan(scan_fn, variable_broadcast='params', variable_carry='cache', out_axes=1, length=self.seq_len)
        _, generated = scan(self, (cond[..., -1], self.make_rng('zdc')))
        return generated
