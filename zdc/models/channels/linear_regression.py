import jax
from sklearn.multioutput import MultiOutputRegressor
from sklearn.linear_model import LinearRegression

from zdc.models.channels.sklearn_utils import sklearn_generate_fn
from zdc.utils.data import load
from zdc.utils.train import train_loop


if __name__ == '__main__':
    key = jax.random.PRNGKey(72)
    init_key, train_key = jax.random.split(key)

    r_train, _, r_test, p_train, _, p_test = load(add_ch_dim=False)

    model = MultiOutputRegressor(LinearRegression(), n_jobs=-1)
    model.fit(p_train, r_train)

    train_loop(
        'linear_regression', None, 'channel', sklearn_generate_fn(model), None, None, (r_test, p_test),
        None, None, None, None, None, train_key, epochs=0
    )
