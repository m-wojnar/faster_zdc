import jax
import jax.numpy as jnp
from flax import linen as nn

from zdc.layers import TransformerBlock, Concatenate


class Transformer(nn.Module):
    vocab_size: int
    embed_dim: int
    seq_len: int
    max_len: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    drop_rate: float
    decode: bool

    def setup(self):
        self.token_emb = nn.Embed(self.vocab_size, self.embed_dim)
        self.cond_emb = nn.Embed(self.vocab_size, self.embed_dim)
        self.pos_emb = nn.Embed(self.max_len, self.embed_dim)
        self.pre_concat = Concatenate(axis=1)
        self.t_blocks = [
            TransformerBlock(self.num_heads, self.hidden_dim, self.drop_rate, self.decode)
            for _ in range(self.num_layers)
        ]
        self.out_ln_x = nn.LayerNorm(use_bias=False, dtype=jnp.bfloat16)
        self.out_ln_c = nn.LayerNorm(use_bias=False, dtype=jnp.bfloat16)
        self.post_concat = Concatenate(axis=1)

    def __call__(self, cond, x, pos, mask, training=True):
        c = self.cond_emb(cond)
        x = self.token_emb(x)
        x = self.pre_concat(c, x)

        pos = self.pos_emb(pos)
        x = x + pos

        for block in self.t_blocks:
            x = block(x, mask, training=training)

        if cond.shape[1] > 0 and x.shape[1] > 0:
            c, x = jnp.split(x, [cond.shape[1] - 1], axis=1)
            c, x = self.out_ln_c(c), self.out_ln_x(x)
        elif cond.shape[1] > 0:
            x = self.out_ln_c(x)
        elif x.shape[1] > 0:
            x = self.out_ln_x(x)

        c = self.cond_emb.attend(c.astype(jnp.float32))
        x = self.token_emb.attend(x.astype(jnp.float32))
        x = self.post_concat(c, x)

        return x


class GPT(nn.Module):
    vocab_size: int
    embed_dim: int
    seq_len: int
    max_len: int
    hidden_dim: int
    num_heads: int
    num_layers: int
    drop_rate: float

    @nn.compact
    def __call__(self, cond, x, training=True):
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
            tokens = jnp.concatenate([cond, x], axis=1)
            pos = jnp.arange(tokens.shape[1])
            mask = nn.make_causal_mask(tokens, dtype=jnp.bfloat16)

        return Transformer(
            self.vocab_size, self.embed_dim, self.seq_len, self.max_len, self.hidden_dim,
            self.num_heads, self.num_layers, self.drop_rate, decode=not training
        )(cond, x, pos, mask, training=training)

    @staticmethod
    def select_top_k(logits, top_k):
        values, indices = jax.lax.top_k(logits, top_k)
        mask = jnp.zeros_like(logits, dtype=bool)
        mask = mask.at[jnp.arange(logits.shape[0])[:, None], indices].set(True)
        return jnp.where(mask, logits, -jnp.inf)

    @staticmethod
    def select_top_p(logits, top_p):
        sorted_logits = jnp.sort(logits, axis=-1)
        sorted_indices = jnp.argsort(logits, axis=-1)
        cumulative_probs = jnp.cumsum(nn.softmax(sorted_logits, axis=-1), axis=-1)
        sorted_logits = jnp.where(cumulative_probs > (1 - top_p), sorted_logits, -jnp.inf)
        out = jnp.empty_like(sorted_logits)
        out = out.at[jnp.arange(logits.shape[0])[:, None], sorted_indices].set(sorted_logits)
        return out

    def gen(self, cond, temperature=1.0, top_k=None, top_p=None):
        def scan_fn(gpt, carry):
            prev_token, key, idx = carry
            key, cat_key = jax.random.split(key)

            input = jax.lax.select(idx < cond.shape[1], cond[:, idx][:, None], prev_token)
            logits = nn.cond(
                idx < cond.shape[1],
                lambda model, x: model(x, empty, training=False),
                lambda model, x: model(empty, x, training=False),
                gpt,input
            )

            if top_k is not None:
                logits = self.select_top_k(logits, top_k)

            if top_p is not None:
                logits = self.select_top_p(logits, top_p)

            next_token = jax.random.categorical(cat_key, logits / temperature)
            return (next_token, key, idx + 1), next_token

        empty = jnp.empty((cond.shape[0], 0), dtype=jnp.int32)
        scan = nn.scan(scan_fn, variable_broadcast='params', variable_carry='cache', out_axes=1, length=self.max_len)
        _, generated = scan(self, (cond[:, -1][:, None], self.make_rng('zdc'), 0))
        return generated[..., -self.seq_len:, 0]
