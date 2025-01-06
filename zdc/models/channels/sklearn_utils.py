from jax import numpy as jnp
from sklearn.metrics import make_scorer
from sklearn.model_selection import GridSearchCV

from zdc.utils.losses import wasserstein_loss


def hyperparameter_search(model, params, r_val, p_val, print_summary=True):
    search = GridSearchCV(model, params, scoring=make_scorer(wasserstein_loss, greater_is_better=False), n_jobs=-1)
    search.fit(p_val, r_val)
    best_params = {k.split('__')[-1]: v for k, v in search.best_params_.items()}

    if print_summary:
        print(f'Best parameters: {best_params}')
        print(f'Best score: {search.best_score_}')

    return best_params


def sklearn_generate_fn(model):
    def generate_fn(params, state, key, *x):
        pred = model.predict(x[1])
        pred = jnp.clip(pred, 0, None)
        return pred.astype(jnp.float32)

    return generate_fn
