"""JAX I2C training on SMAX with vectorised environments."""

from __future__ import annotations

import os
import sys
import time
from pathlib import Path
from typing import Any

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import wandb
from flax import serialization
from flax.training.train_state import TrainState
from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import SMAXLogWrapper

try:
    from baselines.metrics import log_metrics
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from baselines.metrics import log_metrics


class I2CActorCritic(nn.Module):
    action_dim: int
    global_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs: jax.Array, message: jax.Array, global_obs: jax.Array):
        actor_input = jnp.concatenate((obs, message), axis=-1)
        hidden = nn.tanh(nn.Dense(self.hidden_dim)(actor_input))
        hidden = nn.tanh(nn.Dense(self.hidden_dim)(hidden))
        logits = nn.Dense(self.action_dim)(hidden)
        gate_logits = nn.Dense(1)(actor_input).squeeze(-1)
        critic_hidden = nn.tanh(nn.Dense(self.hidden_dim)(global_obs))
        critic_hidden = nn.tanh(nn.Dense(self.hidden_dim)(critic_hidden))
        value = nn.Dense(1)(critic_hidden).squeeze(-1)
        return logits, value, gate_logits


def _message(obs: jax.Array) -> jax.Array:
    """Average the other agents' observations independently per environment."""
    count = obs.shape[0]
    return (jnp.sum(obs, axis=0, keepdims=True) - obs) / max(count - 1, 1)


def _returns(rewards: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
    result = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros((rewards.shape[1],), dtype=np.float32)
    for index in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[index] + gamma * running * (1.0 - dones[index])
        result[index] = running
    return result


def _returns_jax(
    rewards: jax.Array, dones: jax.Array, gamma: float
) -> jax.Array:
    def accumulate(running: jax.Array, transition: tuple[jax.Array, jax.Array]):
        reward, done = transition
        running = reward + gamma * running * (1.0 - done)
        return running, running

    _, reversed_returns = jax.lax.scan(
        accumulate,
        jnp.zeros((rewards.shape[1],), dtype=rewards.dtype),
        (rewards[::-1], dones[::-1]),
    )
    return reversed_returns[::-1]


def make_train(config: dict[str, Any]):
    num_envs = int(config["NUM_ENVS"])
    env = SMAXLogWrapper(HeuristicEnemySMAX(
        scenario=map_name_to_scenario(config["MAP_NAME"]),
        **config["ENV_KWARGS"],
    ))
    agents = env.agents
    num_agents = env.num_agents
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).n
    global_dim = obs_dim * num_agents
    network = I2CActorCritic(action_dim, global_dim, int(config["HIDDEN_DIM"]))
    init_obs = jnp.zeros((num_agents * num_envs, obs_dim))
    init_global = jnp.zeros((num_envs, global_dim))
    state = TrainState.create(
        apply_fn=network.apply,
        params=network.init(
            jax.random.PRNGKey(int(config["SEED"])),
            init_obs,
            init_obs,
            init_global,
        ),
        tx=optax.chain(
            optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
            optax.adam(float(config["LR"])),
        ),
    )
    checkpoint_path = str(config.get("CHECKPOINT_PATH", ""))
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as checkpoint:
            state = serialization.from_bytes(state, checkpoint.read())

    rollout_steps = int(config["NUM_STEPS"])

    @jax.jit
    def train_step(
        state: TrainState,
        rng: jax.Array,
        obs: jax.Array,
        env_state: Any,
    ):
        def rollout_step(carry: tuple[jax.Array, jax.Array, Any], unused: None):
            rng, obs, env_state = carry
            msg = _message(obs)
            global_obs = obs.transpose(1, 0, 2).reshape(num_envs, global_dim)
            flat_obs = obs.reshape(num_agents * num_envs, obs_dim)
            flat_msg = msg.reshape(num_agents * num_envs, obs_dim)
            _, _, gate_logits = network.apply(
                state.params, flat_obs, flat_msg, global_obs
            )
            avail_dict = jax.vmap(env.get_avail_actions)(env_state.env_state)
            avail = jnp.stack([avail_dict[agent] for agent in agents])
            flat_avail = avail.reshape(num_agents * num_envs, action_dim)
            rng, action_key, gate_key, step_key, reset_key = jax.random.split(rng, 5)
            gate_probability = jax.nn.sigmoid(gate_logits)
            gate = (
                jax.random.uniform(gate_key, gate_probability.shape)
                < gate_probability
            ).astype(jnp.float32)
            effective_msg = flat_msg * gate[:, None]
            logits, value, gate_logits = network.apply(
                state.params, flat_obs, effective_msg, global_obs
            )
            logits = logits - (1 - flat_avail) * 1e10
            action = jax.random.categorical(action_key, logits, axis=-1)
            log_probs = jax.nn.log_softmax(logits)
            chosen = jnp.take_along_axis(
                log_probs, action[:, None], axis=-1
            ).squeeze(-1)
            env_actions = {
                agent: action[index * num_envs : (index + 1) * num_envs]
                for index, agent in enumerate(agents)
            }
            next_obs_dict, next_env_state, reward_dict, done_dict, info = jax.vmap(
                env.step, in_axes=(0, 0, 0)
            )(
                jax.random.split(step_key, num_envs),
                env_state,
                env_actions,
            )
            reward = jnp.stack([reward_dict[agent] for agent in agents]).mean(0)
            done = jnp.asarray(done_dict["__all__"], dtype=jnp.float32)
            won = info["returned_won_episode"][:, 0]
            episode = info["returned_episode"][:, 0]
            episode_return = info["returned_episode_returns"][:, 0]
            episode_length = info["returned_episode_lengths"][:, 0]
            reset_obs_dict, reset_env_state = jax.vmap(env.reset)(
                jax.random.split(reset_key, num_envs)
            )
            reset_obs = jnp.stack([reset_obs_dict[agent] for agent in agents])
            done_mask = done.astype(bool)

            def reset_done(reset_value: jax.Array, step_value: jax.Array):
                mask = done_mask.reshape((num_envs,) + (1,) * (step_value.ndim - 1))
                return jnp.where(mask, reset_value, step_value)

            next_obs = jnp.where(
                done_mask[None, :, None],
                reset_obs,
                jnp.stack([next_obs_dict[agent] for agent in agents]),
            )
            next_env_state = jax.tree.map(
                reset_done, reset_env_state, next_env_state
            )
            transition = (
                flat_obs,
                effective_msg,
                global_obs,
                action,
                reward,
                done,
                value,
                gate,
                won,
                episode,
                episode_return,
                episode_length,
            )
            return (rng, next_obs, next_env_state), transition

        (_, next_obs, next_env_state), rollout = jax.lax.scan(
            rollout_step,
            (rng, obs, env_state),
            None,
            length=rollout_steps,
        )
        (
            observations,
            messages,
            globals_,
            actions,
            rewards,
            dones,
            old_values,
            gates,
            won,
            episode,
            episode_return,
            episode_length,
        ) = rollout
        targets = _returns_jax(rewards, dones, float(config["GAMMA"]))
        advantages = targets - old_values

        def loss_fn(params):
            flat_obs = observations.reshape(-1, obs_dim)
            flat_msg = messages.reshape(-1, obs_dim)
            flat_global = globals_.reshape(-1, global_dim)
            logits, values, gate_logits = network.apply(
                params, flat_obs, flat_msg, flat_global
            )
            log_probs = jax.nn.log_softmax(logits)
            selected = jnp.take_along_axis(
                log_probs, actions.reshape(-1, 1), axis=-1
            ).squeeze(-1)
            selected = selected.reshape(rollout_steps, num_agents, num_envs).mean(1)
            entropy = -(jnp.exp(log_probs) * log_probs).sum(-1).mean()
            actor_loss = -(selected * jax.lax.stop_gradient(advantages)).mean()
            critic_loss = jnp.square(
                values.reshape(rollout_steps, num_envs) - targets
            ).mean()
            gate_loss = jnp.mean(
                jax.nn.softplus(gate_logits) - gates.reshape(-1) * gate_logits
            )
            total_loss = (
                actor_loss
                + float(config["VF_COEF"]) * critic_loss
                + 0.01 * gate_loss
                - float(config["ENT_COEF"]) * entropy
            )
            return total_loss, (actor_loss, critic_loss, entropy, gate_loss)

        (_, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(
            state.params
        )
        return (
            state.apply_gradients(grads=grads),
            rng,
            next_obs,
            next_env_state,
            rewards,
            gates,
            won,
            episode,
            episode_return,
            episode_length,
            metrics,
        )

    def train(rng: jax.Array):
        nonlocal state
        rng, reset_key = jax.random.split(rng)
        reset_keys = jax.random.split(reset_key, num_envs)
        obs_dict, env_state = jax.vmap(env.reset)(reset_keys)
        obs = jnp.stack([obs_dict[agent] for agent in agents])
        total_steps = int(config["TOTAL_TIMESTEPS"])
        num_updates = max(
            1, (total_steps + rollout_steps * num_envs - 1) // (rollout_steps * num_envs)
        )
        start_time = time.time()
        for update in range(num_updates):
            (
                state,
                rng,
                obs,
                env_state,
                rewards_arr,
                gates_arr,
                won_arr,
                episode_arr,
                episode_returns_arr,
                episode_lengths_arr,
                metrics,
            ) = train_step(
                state, rng, obs, env_state
            )
            rewards_arr = np.asarray(rewards_arr)
            gates_arr = np.asarray(gates_arr)
            won_arr = np.asarray(won_arr)
            episode_arr = np.asarray(episode_arr)
            episode_returns_arr = np.asarray(episode_returns_arr)
            episode_lengths_arr = np.asarray(episode_lengths_arr)
            completed = episode_arr.astype(bool)
            completed_episodes = int(completed.sum())
            completed_wins = int(won_arr[completed].sum())
            timestep = min((update + 1) * rollout_steps * num_envs, total_steps)
            log_metrics(
                {
                    "returned_episode_returns": float(
                        episode_returns_arr[completed].mean()
                    )
                    if completed_episodes
                    else float("nan"),
                    "returned_won_episode": float(
                        won_arr[completed].mean()
                    )
                    if completed_episodes
                    else float("nan"),
                    "returned_episode": float(
                        episode_arr[completed].mean()
                    )
                    if completed_episodes
                    else float("nan"),
                    "returned_episode_lengths": float(
                        episode_lengths_arr[completed].mean()
                    )
                    if completed_episodes
                    else float("nan"),
                    "policy/communication_rate": float(gates_arr.mean()),
                    "episodes/completed": completed_episodes,
                    "episodes/wins": completed_wins,
                    "training/update": update + 1,
                    "training/timesteps": timestep,
                    "training/updates_per_second": (update + 1) / max(time.time() - start_time, 1e-6),
                },
                step=timestep,
            )
            print(
                f"I2C SMAX update {update + 1}/{num_updates} "
                f"reward={rewards_arr.mean():.4f} comm={gates_arr.mean():.3f}",
                flush=True,
            )
            if checkpoint_path and (update + 1) % int(config["CHECKPOINT_INTERVAL"]) == 0:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "wb") as checkpoint:
                    checkpoint.write(serialization.to_bytes(state))
        return {"runner_state": state}

    return train


def main() -> None:
    config: dict[str, Any] = {
        "MAP_NAME": "3s5z",
        "ENV_KWARGS": {
            "see_enemy_actions": True,
            "walls_cause_death": True,
            "attack_mode": "closest",
        },
        "NUM_ENVS": 16,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1_000_000,
        "SEED": 0,
        "LR": 3e-4,
        "GAMMA": 0.99,
        "ENT_COEF": 0.01,
        "VF_COEF": 0.5,
        "MAX_GRAD_NORM": 0.5,
        "HIDDEN_DIM": 128,
        "PROJECT": "jaxmarl",
        "ENTITY": "",
        "WANDB_MODE": "disabled",
        "CHECKPOINT_PATH": "",
        "CHECKPOINT_INTERVAL": 1,
    }
    int_keys = {"SEED", "NUM_ENVS", "NUM_STEPS", "TOTAL_TIMESTEPS", "CHECKPOINT_INTERVAL", "HIDDEN_DIM"}
    float_keys = {"LR", "GAMMA", "ENT_COEF", "VF_COEF", "MAX_GRAD_NORM"}
    string_keys = {"PROJECT", "ENTITY", "WANDB_MODE", "CHECKPOINT_PATH", "MAP_NAME"}
    for argument in sys.argv[1:]:
        if "=" not in argument:
            continue
        key, value = argument.lstrip("+").split("=", 1)
        key = key[4:] if key.startswith("alg.") else key
        if key in int_keys:
            config[key] = int(value)
        elif key in float_keys:
            config[key] = float(value)
        elif key in string_keys:
            config[key] = value
    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        mode=config["WANDB_MODE"],
        tags=["I2C", "SMAX", config["MAP_NAME"]],
        group=os.environ.get("WANDB_RUN_GROUP") or None,
        name=os.environ.get("WANDB_NAME") or None,
        config=config,
    )
    make_train(config)(jax.random.PRNGKey(config["SEED"]))
    run.finish()


if __name__ == "__main__":
    main()
