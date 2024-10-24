from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import optuna

from zdc.models.autoencoder.variational import VAE, Discriminator, disc_loss_fn, gen_loss_fn, step_fn
from zdc.utils.data import load, batches
from zdc.utils.losses import perceptual_loss
from zdc.utils.nn import opt_with_cosine_schedule, init, forward
from zdc.utils.train import default_eval_fn


def suggest_optimizer(trial, epochs, batch_size, prefix):
    learning_rate = trial.suggest_float(f'{prefix}_learning_rate', 1e-6, 1e-2, log=True)
    beta_1 = trial.suggest_float(f'{prefix}_beta_1', 0.4, 1.)
    beta_2 = trial.suggest_float(f'{prefix}_beta_2', 0.4, 1.)
    optimizer = partial(optax.adam, b1=beta_1, b2=beta_2)

    if trial.suggest_categorical(f'{prefix}_use_cosine_decay', [True, False]):
        return opt_with_cosine_schedule(optimizer, learning_rate, epochs=epochs, batch_size=batch_size)
    else:
        return optimizer(learning_rate)


def objective(trial, train_dataset, val_dataset, n_rep=5, epochs=50, batch_size=256):
    gen_optimizer = suggest_optimizer(trial, epochs, batch_size, 'gen')
    disc_optimizer = suggest_optimizer(trial, epochs, batch_size, 'disc')

    seed = np.random.randint(0, 2**32 - 1)
    gen_init_key, disc_init_key, train_key, val_key, shuffle_key = jax.random.split(jax.random.PRNGKey(seed), 5)

    gen_model = VAE()
    gen_params, gen_state = init(gen_model, gen_init_key, r_train[:5])
    gen_opt_state = gen_optimizer.init(gen_params)

    disc_model = Discriminator()
    disc_params, disc_state = init(disc_model, disc_init_key, r_train[:5])
    disc_opt_state = disc_optimizer.init(disc_params)

    params = (gen_params, disc_params)
    state = (gen_state, disc_state)
    opt_state = (gen_opt_state, disc_opt_state)

    train_fn = jax.jit(partial(
        step_fn,
        disc_optimizer=disc_optimizer,
        gen_optimizer=gen_optimizer,
        disc_loss_fn=partial(disc_loss_fn, gen_model=gen_model, disc_model=disc_model),
        gen_loss_fn=partial(gen_loss_fn, gen_model=gen_model, disc_model=disc_model, lpips_fn=perceptual_loss())
    ))
    generate_fn = jax.jit(lambda params, state, key, *x: forward(gen_model, params[0], state[0], key, x[0])[0][0])

    for _ in range(epochs):
        shuffle_key, shuffle_train_subkey = jax.random.split(shuffle_key)

        for batch in batches(*train_dataset, batch_size=batch_size, shuffle_key=shuffle_train_subkey):
            train_key, subkey = jax.random.split(train_key)
            params, opt_state, (state, *losses) = train_fn(params, (state, subkey, *batch), opt_state)

    generated, original = [], []

    for batch in batches(*val_dataset, batch_size=batch_size):
        for _ in range(n_rep):
            val_key, subkey = jax.random.split(val_key)
            generated.append(generate_fn(params, state, subkey, *batch))
            original.append(batch)

    generated, original = jnp.concatenate(generated), (jnp.concatenate(xs) for xs in zip(*original))
    _, _, val_wasserstein = default_eval_fn(generated, *original)

    return val_wasserstein


if __name__ == "__main__":
    args = ArgumentParser()
    args.add_argument('--database', required=True, type=str)
    args.add_argument('--name', required=True, type=str)
    args.add_argument('--trials', type=int, default=200)
    args = args.parse_args()

    r_train, r_val, _, p_train, p_val, _ = load()

    study = optuna.create_study(
        storage=args.database,
        study_name=args.name,
        load_if_exists=True,
        direction='minimize',
        sampler=optuna.samplers.TPESampler()
    )

    study.optimize(
        partial(objective, train_dataset=(r_train, p_train), val_dataset=(r_val, p_val)),
        n_trials=args.trials,
        gc_after_trial=True
    )
