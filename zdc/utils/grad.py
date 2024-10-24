import jax
import optax


@jax.custom_vjp
def grad_norm(x, weight):
    return x


def grad_norm_fwd(x, weight):
    return x, weight


def grad_norm_bwd(weight, g):
    gn = optax.tree_utils.tree_l2_norm(g)
    g = weight * g / (gn + 1e-8)
    return g, None


grad_norm.defvjp(grad_norm_fwd, grad_norm_bwd)
