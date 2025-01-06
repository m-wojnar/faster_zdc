import jax
from sklearn.multioutput import MultiOutputRegressor
from sklearn.neighbors import KNeighborsRegressor

from zdc.models.channels.sklearn_utils import hyperparameter_search, sklearn_generate_fn
from zdc.utils.data import load
from zdc.utils.train import train_loop


if __name__ == '__main__':
    key = jax.random.PRNGKey(72)
    init_key, train_key = jax.random.split(key)

    r_train, r_val, r_test, p_train, p_val, p_test = load(add_ch_dim=False)

    params = hyperparameter_search(
        model=MultiOutputRegressor(KNeighborsRegressor()),
        params={'estimator__n_neighbors': [3, 5, 10, 15, 20, 25]},
        r_val=r_val,
        p_val=p_val
    )
    model = MultiOutputRegressor(KNeighborsRegressor(**params), n_jobs=-1)
    model.fit(p_train, r_train)

    train_loop(
        'knn', None, 'channel', sklearn_generate_fn(model), None, None, (r_test, p_test),
        None, None, None, None, None, train_key, epochs=0
    )
