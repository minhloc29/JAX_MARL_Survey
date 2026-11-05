"""A small, fully JAX/Flax FACMAC implementation.

The implementation deliberately keeps the environment-facing code free of
framework-specific adapters.  Discrete SMAX uses categorical policies while
MABRAX uses squashed Gaussian policies; both share the same centralised critic
and factorised actor interface.
"""

from __future__ import annotations

import sys
import os
import time
from typing import Any

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax
import wandb
from flax import serialization
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
DictConfig = dict

from jaxmarl import make
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.environments import spaces


class FACMACActor(nn.Module):
    """Parameter-sharing factorised actor (one output row per agent)."""

    output_dim: int
    continuous: bool = False
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs: jax.Array) -> jax.Array:
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(obs)
        x = nn.tanh(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(x)
        x = nn.tanh(x)
        return nn.Dense(self.output_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)


class FACMACCritic(nn.Module):
    """Centralised mixing critic, conditioned on global observations/actions."""

    hidden_dim: int = 256

    @nn.compact
    def __call__(self, state: jax.Array, actions: jax.Array) -> jax.Array:
        x = jnp.concatenate((state.reshape(state.shape[:-2] + (-1,)),
                             actions.reshape(actions.shape[:-2] + (-1,))), axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(1.0))(x)
        x = nn.relu(x)
        return nn.Dense(1, kernel_init=orthogonal(1.0))(x).squeeze(-1)


def _stack(mapping: dict[str, jax.Array], agents: list[str], width: int | None = None) -> jax.Array:
    values = [jnp.asarray(mapping[a]) for a in agents]
    if width is not None:
        values = [jnp.pad(v, [(0, 0)] * (v.ndim - 1) + [(0, width - v.shape[-1])]) for v in values]
    return jnp.stack(values)


def _global_norm(tree: Any) -> jax.Array:
    leaves = jax.tree_util.tree_leaves(tree)
    return jnp.sqrt(sum(jnp.sum(jnp.square(leaf)) for leaf in leaves))


def _save_checkpoint(path: str, actor_state: TrainState, critic_state: TrainState, update: int) -> None:
    checkpoint = {
        "actor_state": actor_state,
        "critic_state": critic_state,
        "update": update,
    }
    destination = os.path.abspath(path)
    os.makedirs(os.path.dirname(destination) or ".", exist_ok=True)
    with open(destination, "wb") as file:
        file.write(serialization.to_bytes(checkpoint))


def _load_checkpoint(path: str, actor_state: TrainState, critic_state: TrainState):
    with open(path, "rb") as file:
        checkpoint = serialization.from_bytes(
            {
                "actor_state": actor_state,
                "critic_state": critic_state,
                "update": 0,
            },
            file.read(),
        )
    return checkpoint["actor_state"], checkpoint["critic_state"], int(checkpoint["update"])


def make_train(config: DictConfig | dict[str, Any]):
    """Build a jittable FACMAC training callable.

    This is intentionally a compact baseline suitable for smoke tests and
    extensions: the critic is centralised and the actor remains factorised.
    """
    cfg = dict(config)
    env_name = cfg.get("ENV_NAME", "SMAX")
    env_kwargs = dict(cfg.get("ENV_KWARGS", {}))
    if env_name == "SMAX":
        env_kwargs["scenario"] = map_name_to_scenario(cfg.get("MAP_NAME", "3s5z"))
    env = make(env_name, **env_kwargs)
    agents = list(env.agents)
    obs_dim = max(env.observation_space(a).shape[-1] for a in agents)
    action_spaces = [env.action_space(a) for a in agents]
    continuous = isinstance(action_spaces[0], spaces.Box)
    action_dim = max((s.shape[-1] if continuous else s.n) for s in action_spaces)
    actor, critic = FACMACActor(action_dim, continuous), FACMACCritic()

    def train(rng):
        rng, k1, k2 = jax.random.split(rng, 3)
        obs, env_state = env.reset(k1)
        obs0 = _stack(obs, agents, obs_dim)
        state0 = obs0.reshape(-1)
        act0 = jnp.zeros((len(agents), action_dim))
        tx = optax.adam(float(cfg.get("LR", 3e-4)))
        actor_state = TrainState.create(
            apply_fn=actor.apply, params=actor.init(k2, obs0), tx=tx
        )
        critic_state = TrainState.create(
            apply_fn=critic.apply, params=critic.init(k2, state0, act0), tx=tx
        )
        checkpoint_path = cfg.get("CHECKPOINT_PATH") or None
        start_update = 0
        if checkpoint_path:
            if os.path.exists(checkpoint_path):
                actor_state, critic_state, start_update = _load_checkpoint(
                    checkpoint_path, actor_state, critic_state
                )
                print(
                    f"Resumed FACMAC checkpoint {checkpoint_path} at update {start_update}",
                    flush=True,
                )
            else:
                print(f"Starting new FACMAC checkpoint at {checkpoint_path}", flush=True)

        def critic_actions(action):
            if continuous:
                return action
            return jax.nn.one_hot(action, action_dim).reshape((len(agents), action_dim))

        def actor_actions(params, x, key):
            output = actor.apply(params, x)
            if continuous:
                return jnp.tanh(output), output
            return jax.random.categorical(key, output, axis=-1), output

        def step(carry, _):
            actor_state, critic_state, env_state, obs, rng = carry
            x = _stack(obs, agents, obs_dim)
            rng, ka, ks = jax.random.split(rng, 3)
            action, _ = actor_actions(actor_state.params, x, ka)
            env_action = {a: action[i, : (action_spaces[i].shape[-1] if continuous else 1)]
                          if continuous else action[i] for i, a in enumerate(agents)}
            next_obs, next_state, reward, done, info = env.step(ks, env_state, env_action)
            r = jnp.mean(jnp.stack([jnp.asarray(reward[a]) for a in agents]))
            d = jnp.asarray(done["__all__"])
            return (actor_state, critic_state, next_state, next_obs, rng), (x, action, r, d)

        rollout = int(cfg.get("NUM_STEPS", 128))
        total = int(cfg.get("TOTAL_TIMESTEPS", rollout))
        if rollout < 1 or total < 1:
            raise ValueError("NUM_STEPS and TOTAL_TIMESTEPS must be positive")
        updates = (total + rollout - 1) // rollout
        metric_values = []
        checkpoint_interval = int(cfg.get("CHECKPOINT_INTERVAL", 1))
        for update in range(start_update, updates):
            update_start = time.monotonic()
            horizon = min(rollout, total - update * rollout)
            print(
                f"FACMAC update {update + 1}/{updates} "
                f"(timesteps {(update * rollout) + horizon}/{total})",
                flush=True,
            )
            (actor_state, critic_state, env_state, obs, rng), batch = jax.lax.scan(
                step, (actor_state, critic_state, env_state, obs, rng), None, horizon
            )
            observations, actions, rewards, dones = batch
            next_x = _stack(obs, agents, obs_dim)
            states = observations.reshape((horizon, -1))
            action_features = jax.vmap(critic_actions)(actions)
            next_action, _ = actor_actions(actor_state.params, next_x, rng)
            next_features = critic_actions(next_action)
            next_value = critic_state.apply_fn(
                critic_state.params, next_x.reshape(-1), next_features
            )
            targets = rewards + float(cfg.get("GAMMA", 0.99)) * (1.0 - dones) * next_value

            def critic_loss(params):
                values = jax.vmap(critic_state.apply_fn, in_axes=(None, 0, 0))(
                    params, states, action_features
                )
                return jnp.mean((values - jax.lax.stop_gradient(targets)) ** 2)

            values_before = jax.vmap(critic_state.apply_fn, in_axes=(None, 0, 0))(
                critic_state.params, states, action_features
            )
            critic_loss_value, critic_grads = jax.value_and_grad(critic_loss)(
                critic_state.params
            )
            critic_grad_norm = _global_norm(critic_grads)
            critic_state = critic_state.apply_gradients(grads=critic_grads)

            def actor_loss(params):
                output = jax.vmap(actor.apply, in_axes=(None, 0))(params, observations)
                if continuous:
                    policy_actions = jnp.tanh(output)
                    log_prob = -jnp.sum((policy_actions - actions) ** 2, axis=(-1, -2))
                else:
                    # The discrete actor uses the sampled rollout action for the
                    # score-function estimator; the critic remains differentiable
                    # for continuous actions.
                    log_prob = jax.nn.log_softmax(output, axis=-1)
                    log_prob = jnp.take_along_axis(
                        log_prob, actions[..., None], axis=-1
                    ).squeeze(-1)
                    policy_actions = actions
                    log_prob = jnp.sum(log_prob, axis=-1)
                features = jax.vmap(critic_actions)(policy_actions)
                q = jax.vmap(critic_state.apply_fn, in_axes=(None, 0, 0))(
                    critic_state.params, states, features
                )
                return -jnp.mean(log_prob * jax.lax.stop_gradient(q))

            actor_loss_value, actor_grads = jax.value_and_grad(actor_loss)(
                actor_state.params
            )
            actor_grad_norm = _global_norm(actor_grads)
            actor_state = actor_state.apply_gradients(grads=actor_grads)
            actor_output = jax.vmap(actor.apply, in_axes=(None, 0))(
                actor_state.params, observations
            )
            entropy = (
                -jnp.mean(
                    jnp.sum(
                        jax.nn.softmax(actor_output)
                        * jax.nn.log_softmax(actor_output),
                        axis=-1,
                    )
                )
                if not continuous
                else jnp.array(0.0)
            )
            timesteps = min((update + 1) * rollout, total)
            metrics = {
                "loss/critic": float(critic_loss_value),
                "loss/actor": float(actor_loss_value),
                "loss/td_abs": float(jnp.mean(jnp.abs(values_before - targets))),
                "q/value_mean": float(jnp.mean(values_before)),
                "q/value_max": float(jnp.max(values_before)),
                "q/target_mean": float(jnp.mean(targets)),
                "q/target_max": float(jnp.max(targets)),
                "q/advantage_mean": float(jnp.mean(targets - values_before)),
                "grad/critic_norm": float(critic_grad_norm),
                "grad/actor_norm": float(actor_grad_norm),
                "policy/action_mean": float(jnp.mean(actions)),
                "policy/action_std": float(jnp.std(actions)),
                "policy/entropy": float(entropy),
                "rollout/reward_mean": float(jnp.mean(rewards)),
                "rollout/reward_std": float(jnp.std(rewards)),
                "rollout/reward_min": float(jnp.min(rewards)),
                "rollout/reward_max": float(jnp.max(rewards)),
                "rollout/return": float(jnp.sum(rewards)),
                "rollout/done_rate": float(jnp.mean(dones)),
                "timesteps": timesteps,
                "update": update + 1,
                "time/update_seconds": time.monotonic() - update_start,
            }
            wandb.log(
                metrics,
                step=timesteps,
            )
            print(
                "  "
                f"reward={metrics['rollout/reward_mean']:.4f} "
                f"return={metrics['rollout/return']:.4f} "
                f"critic={metrics['loss/critic']:.4f} "
                f"actor={metrics['loss/actor']:.4f} "
                f"td={metrics['loss/td_abs']:.4f}",
                flush=True,
            )
            metric_values.append((critic_loss_value, actor_loss_value))
            if checkpoint_path and (update + 1) % checkpoint_interval == 0:
                _save_checkpoint(checkpoint_path, actor_state, critic_state, update + 1)
        if not metric_values:
            return {
                "metrics": {
                    "critic_loss": jnp.empty((0,)),
                    "actor_loss": jnp.empty((0,)),
                },
                "runner_state": (actor_state, critic_state),
            }
        critic_losses, actor_losses = zip(*metric_values)
        return {
            "metrics": {
                "critic_loss": jnp.stack(critic_losses),
                "actor_loss": jnp.stack(actor_losses),
            },
            "runner_state": (actor_state, critic_state),
        }

    return train


def _main(config: DictConfig) -> None:
    """Hydra entry point used by the common baseline launcher."""
    run = wandb.init(
        entity=config.get("ENTITY", ""),
        project=config.get("PROJECT", "jaxmarl"),
        mode=config.get("WANDB_MODE", "disabled"),
        tags=[
            "FACMAC",
            str(config.get("ENV_NAME", "SMAX")),
            *[tag for tag in os.environ.get("WANDB_TAGS", "").split(",") if tag],
        ],
        group=os.environ.get("WANDB_RUN_GROUP") or None,
        name=os.environ.get("WANDB_NAME") or None,
        config=config,
    )
    print(
        f"Starting JAX FACMAC: project={run.project} run={run.name} "
        f"budget={config.get('TOTAL_TIMESTEPS')} rollout={config.get('NUM_STEPS')}",
        flush=True,
    )
    train = make_train(config)
    result = train(jax.random.PRNGKey(int(config.get("SEED", 0))))
    print("FACMAC JAX training complete:", result["metrics"], flush=True)
    run.finish()


def main() -> None:
    config: dict[str, Any] = {
        "ENV_NAME": "SMAX",
        "MAP_NAME": "3s5z",
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1_000_000,
        "SEED": 0,
        "PROJECT": "jaxmarl",
        "ENTITY": "",
        "WANDB_MODE": "disabled",
        "CHECKPOINT_PATH": "",
        "CHECKPOINT_INTERVAL": 1,
    }
    for argument in sys.argv[1:]:
        if "=" not in argument:
            continue
        key, value = argument.lstrip("+").split("=", 1)
        if key == "alg":
            config["ENV_NAME"] = "SMAX" if "smac" in value else "ant_4x2"
        elif key.startswith("alg."):
            key = key[4:]
            if key in {"NUM_STEPS", "TOTAL_TIMESTEPS", "SEED"}:
                value = int(value)
            config[key] = value
        elif key in {"SEED", "NUM_STEPS", "TOTAL_TIMESTEPS", "CHECKPOINT_INTERVAL"}:
            config[key] = int(value)
        elif key in {"PROJECT", "ENTITY", "WANDB_MODE", "CHECKPOINT_PATH"}:
            config[key] = value
    _main(config)


if __name__ == "__main__":
    main()
