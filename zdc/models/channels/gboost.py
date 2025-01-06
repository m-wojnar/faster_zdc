import jax
from sklearn.multioutput import MultiOutputRegressor
from sklearn.ensemble import GradientBoostingRegressor

from zdc.models.channels.sklearn_utils import hyperparameter_search, sklearn_generate_fn
from zdc.utils.data import load
from zdc.utils.train import train_loop


if __name__ == '__main__':
    key = jax.random.PRNGKey(72)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load(add_ch_dim=False)

    params = hyperparameter_search(
        model=MultiOutputRegressor(GradientBoostingRegressor()),
        params={
            'estimator__n_estimators': [100, 200, 300, 400],
            'estimator__max_depth': [2, 3, 4, 5, 6],
            'estimator__learning_rate': [1e-2, 5e-2, 1e-1, 5e-1, 1]
        },
        r_val=r_val,
        p_val=p_val
    )
    model = MultiOutputRegressor(GradientBoostingRegressor(**params), n_jobs=-1)
    model.fit(p_train, r_train)

    train_loop(
        'gboost', None, 'channel', sklearn_generate_fn(model), None, None, (r_test, p_test),
        None, None, None, None, None, train_key, epochs=0
    )
