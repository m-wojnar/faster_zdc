from argparse import ArgumentParser
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax
import optuna

from zdc.models.autoencoder.variational import Model, disc_loss_fn, gen_loss_fn, step_fn
from zdc.utils.data import load, batches
from zdc.utils.losses import perceptual_loss
from zdc.utils.nn import opt_with_cosine_schedule, init, forward, get_layers
from zdc.utils.train import default_eval_fn


def suggest_optimizer(trial, epochs, batch_size, prefix):
    learning_rate = trial.suggest_float(f'{prefix}_learning_rate', 5e-6, 5e-3, log=True)
    beta_1 = trial.suggest_float(f'{prefix}_beta_1', 0.5, 1.)
    beta_2 = trial.suggest_float(f'{prefix}_beta_2', 0.5, 1.)
    optimizer = partial(optax.adam, b1=beta_1, b2=beta_2)

    if trial.suggest_categorical(f'{prefix}_use_cosine_decay', [True, False]):
        return opt_with_cosine_schedule(optimizer, learning_rate, epochs=epochs, batch_size=batch_size)
    else:
        return optimizer(learning_rate)


def objective(trial, train_dataset, val_dataset, n_rep=5, epochs=50, batch_size=256):
    gen_optimizer = suggest_optimizer(trial, epochs, batch_size, 'gen')
    disc_optimizer = suggest_optimizer(trial, epochs, batch_size, 'disc')

    kl_weight = trial.suggest_float('kl_weight', 0.01, 1.0, log=True)
    adv_weight = trial.suggest_float('adv_weight', 0.001, 10.0, log=True)
    perc_weight = trial.suggest_float('perc_weight', 0.001, 10.0, log=True)

    seed = np.random.randint(0, 2**32 - 1)
    init_key, train_key, val_key, shuffle_key = jax.random.split(jax.random.PRNGKey(seed), 4)

    model = Model()
    params, state = init(model, init_key, r_train[:5], r_train[:5])
    disc_opt_state = disc_optimizer.init(get_layers(params, 'discriminator'))
    gen_opt_state = gen_optimizer.init(get_layers(params, 'generator'))
    opt_state = (disc_opt_state, gen_opt_state)

    train_fn = jax.jit(partial(
        step_fn,
        disc_optimizer=disc_optimizer,
        gen_optimizer=gen_optimizer,
        disc_loss_fn=partial(disc_loss_fn, model=model),
        gen_loss_fn=partial(gen_loss_fn, model=model, loss_weights=(1.0, kl_weight, adv_weight, perc_weight), lpips_fn=perceptual_loss())
    ))
    generate_fn = jax.jit(lambda params, state, key, *x: forward(model, params, state, key, x[0], method='gen')[0])

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
    args.add_argument('--trials', required=False, type=int, default=200)
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
