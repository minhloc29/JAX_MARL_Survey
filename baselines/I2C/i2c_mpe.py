"""JAX implementation of Individually Inferred Communication (I2C).

The original implementation uses TensorFlow 1 and MADDPG for MPE.  This
version keeps the central I2C mechanism (a learned per-agent communication
gate and message encoder) while using a fixed-shape, discrete actor-critic
for JaxMARL's functional MPE environments.
"""

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
from jaxmarl import make
from jaxmarl.wrappers.baselines import MPELogWrapper


class I2CActorCritic(nn.Module):
    action_dim: int
    num_agents: int
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


def _discounted_returns(rewards: np.ndarray, dones: np.ndarray, gamma: float) -> np.ndarray:
    returns = np.zeros_like(rewards, dtype=np.float32)
    running = np.zeros((rewards.shape[1],), dtype=np.float32)
    for index in range(rewards.shape[0] - 1, -1, -1):
        running = rewards[index] + gamma * running * (1.0 - dones[index])
        returns[index] = running
    return returns


def _message(obs: jax.Array) -> jax.Array:
    """Build the fixed-shape message input from other agents' observations."""
    count = obs.shape[0]
    other_sum = jnp.sum(obs, axis=0, keepdims=True) - obs
    return other_sum / max(count - 1, 1)


def make_train(config: dict[str, Any]):
    env = MPELogWrapper(make(config["ENV_NAME"], **config.get("ENV_KWARGS", {})))
    agents = env.agents
    num_agents = env.num_agents
    obs_dim = env.observation_space(agents[0]).shape[0]
    action_dim = env.action_space(agents[0]).n
    global_dim = obs_dim * num_agents
    network = I2CActorCritic(action_dim, num_agents, int(config["HIDDEN_DIM"]))
    init_obs = jnp.zeros((num_agents, obs_dim))
    init_global = jnp.zeros((global_dim,))
    params = network.init(
        jax.random.PRNGKey(int(config["SEED"])), init_obs, init_obs, init_global
    )
    tx = optax.chain(
        optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
        optax.adam(float(config["LR"])),
    )
    state = TrainState.create(apply_fn=network.apply, params=params, tx=tx)
    checkpoint_path = str(config.get("CHECKPOINT_PATH", ""))
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as checkpoint:
            state = serialization.from_bytes(state, checkpoint.read())

    def train(rng: jax.Array):
        nonlocal state
        reset_key, rng = jax.random.split(rng)
        obs_dict, env_state = env.reset(reset_key)
        obs = jnp.stack([obs_dict[agent] for agent in agents])
        rollout_steps = int(config["NUM_STEPS"])
        total_steps = int(config["TOTAL_TIMESTEPS"])
        num_updates = max(1, (total_steps + rollout_steps - 1) // rollout_steps)
        start_time = time.time()
        metrics_history: list[dict[str, float]] = []

        for update in range(num_updates):
            obs_batch, msg_batch, global_batch = [], [], []
            action_batch, reward_batch, done_batch = [], [], []
            log_prob_batch, value_batch, gate_batch = [], [], []
            episode_returns = []
            current_return = np.zeros((num_agents,), dtype=np.float32)

            for _ in range(rollout_steps):
                msg = _message(obs)
                global_obs = obs.reshape(-1)
                logits, value, gate_logits = network.apply(
                    state.params, obs, msg, global_obs
                )
                rng, action_key, gate_key = jax.random.split(rng, 3)
                gate = jax.nn.sigmoid(gate_logits)
                gate = jnp.where(
                    jax.random.uniform(gate_key, gate.shape) < gate, 1.0, 0.0
                )
                effective_msg = msg * gate[:, None]
                logits, value, gate_logits = network.apply(
                    state.params, obs, effective_msg, global_obs
                )
                actions = jax.random.categorical(action_key, logits, axis=-1)
                log_probs = jax.nn.log_softmax(logits)
                chosen_log_probs = jnp.take_along_axis(
                    log_probs, actions[:, None], axis=-1
                ).squeeze(-1)
                env_actions = {agent: actions[index] for index, agent in enumerate(agents)}
                step_key, rng = jax.random.split(rng)
                next_obs_dict, env_state, rewards, dones, info = env.step(
                    step_key, env_state, env_actions
                )
                next_obs = jnp.stack([next_obs_dict[agent] for agent in agents])
                reward = jnp.stack([rewards[agent] for agent in agents])
                done = jnp.asarray(dones["__all__"], dtype=jnp.float32)
                obs_batch.append(obs)
                msg_batch.append(effective_msg)
                global_batch.append(global_obs)
                action_batch.append(actions)
                reward_batch.append(reward)
                done_batch.append(jnp.full((num_agents,), done))
                log_prob_batch.append(chosen_log_probs)
                value_batch.append(value)
                gate_batch.append(gate)
                current_return += np.asarray(reward)
                if bool(done):
                    episode_returns.append(float(np.mean(current_return)))
                    current_return.fill(0)
                obs = next_obs

            obs_arr = jnp.stack(obs_batch)
            msg_arr = jnp.stack(msg_batch)
            global_arr = jnp.stack(global_batch)
            actions_arr = jnp.stack(action_batch)
            rewards_arr = np.asarray(jnp.stack(reward_batch))
            dones_arr = np.asarray(jnp.stack(done_batch))
            old_log_probs = jnp.stack(log_prob_batch)
            old_values = jnp.stack(value_batch)
            gates = jnp.stack(gate_batch)
            returns = jnp.asarray(_discounted_returns(rewards_arr, dones_arr, float(config["GAMMA"])))
            global_returns = returns.mean(axis=1)
            advantages = (global_returns - old_values)[:, None]
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            def loss_fn(params):
                logits, values, gate_logits = network.apply(
                    params, obs_arr, msg_arr, global_arr
                )
                log_probs = jax.nn.log_softmax(logits)
                selected = jnp.take_along_axis(
                    log_probs, actions_arr[..., None], axis=-1
                ).squeeze(-1)
                entropy = -(jnp.exp(log_probs) * log_probs).sum(-1).mean()
                actor_loss = -(selected * jax.lax.stop_gradient(advantages)).mean()
                critic_loss = jnp.mean(jnp.square(values - global_returns))
                gate_loss = jnp.mean(jax.nn.softplus(gate_logits) - gates * gate_logits)
                total = (
                    actor_loss
                    + float(config["VF_COEF"]) * critic_loss
                    + 0.01 * gate_loss
                    - float(config["ENT_COEF"]) * entropy
                )
                return total, (actor_loss, critic_loss, entropy, gate_loss, values, gate_logits)

            (total_loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(state.params)
            state = state.apply_gradients(grads=grads)
            actor_loss, critic_loss, entropy, gate_loss, values, gate_logits = aux
            elapsed = time.time() - start_time
            metrics = {
                "loss/total": float(total_loss),
                "loss/actor": float(actor_loss),
                "loss/critic": float(critic_loss),
                "loss/communication_gate": float(gate_loss),
                "policy/entropy": float(entropy),
                "policy/communication_rate": float(gates.mean()),
                "value/mean": float(values.mean()),
                "value/target_mean": float(returns.mean()),
                "rollout/reward_mean": float(rewards_arr.mean()),
                "rollout/reward_std": float(rewards_arr.std()),
                "rollout/return": float(rewards_arr.sum(axis=0).mean()),
                "rollout/done_rate": float(dones_arr.mean()),
                "training/update": update + 1,
                "training/timesteps": min((update + 1) * rollout_steps, total_steps),
                "training/updates_per_second": (update + 1) / max(elapsed, 1e-6),
            }
            if episode_returns:
                metrics["evaluation/episode_return"] = float(np.mean(episode_returns))
            wandb.log(metrics, step=metrics["training/timesteps"])
            metrics_history.append(metrics)
            print(
                f"I2C update {update + 1}/{num_updates} "
                f"reward={metrics['rollout/reward_mean']:.4f} "
                f"return={metrics['rollout/return']:.4f} "
                f"comm={metrics['policy/communication_rate']:.3f}",
                flush=True,
            )
            if checkpoint_path and (update + 1) % int(config["CHECKPOINT_INTERVAL"]) == 0:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "wb") as checkpoint:
                    checkpoint.write(serialization.to_bytes(state))
        return {"metrics": metrics_history, "runner_state": state}

    return train


def main() -> None:
    config: dict[str, Any] = {
        "ENV_NAME": "MPE_simple_spread_v3",
        "ENV_KWARGS": {},
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
    for argument in sys.argv[1:]:
        if "=" not in argument:
            continue
        key, value = argument.lstrip("+").split("=", 1)
        key = key[4:] if key.startswith("alg.") else key
        if key == "alg":
            continue
        if key in {"SEED", "NUM_STEPS", "TOTAL_TIMESTEPS", "CHECKPOINT_INTERVAL", "HIDDEN_DIM"}:
            config[key] = int(value)
        elif key in {"LR", "GAMMA", "ENT_COEF", "VF_COEF", "MAX_GRAD_NORM"}:
            config[key] = float(value)
        elif key in {"PROJECT", "ENTITY", "WANDB_MODE", "CHECKPOINT_PATH", "ENV_NAME"}:
            config[key] = value
    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        mode=config["WANDB_MODE"],
        tags=["I2C", config["ENV_NAME"]],
        group=os.environ.get("WANDB_RUN_GROUP") or None,
        name=os.environ.get("WANDB_NAME") or None,
        config=config,
    )
    print(
        f"Starting JAX I2C: project={run.project} run={run.name} "
        f"budget={config['TOTAL_TIMESTEPS']} rollout={config['NUM_STEPS']}",
        flush=True,
    )
    make_train(config)(jax.random.PRNGKey(config["SEED"]))
    run.finish()


if __name__ == "__main__":
    main()
