"""GPU-batched JAX implementation of NDQ on SMAX.

The original NDQ implementation is PyMARL/Torch based. This implementation
keeps recurrent agent Q-values, QMIX mixing, and stochastic communication
regularisation while compiling rollout and update work with JAX.
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
from jaxmarl.environments.smax import map_name_to_scenario
from jaxmarl.wrappers.baselines import CTRolloutManager, SMAXLogWrapper

try:
    from baselines.metrics import finished_episode_stats, log_metrics
except ModuleNotFoundError:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from baselines.metrics import finished_episode_stats, log_metrics


class AgentQNetwork(nn.Module):
    action_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, hidden, obs, message):
        x = nn.relu(nn.Dense(self.hidden_dim)(jnp.concatenate((obs, message), -1)))
        hidden, x = nn.GRUCell(self.hidden_dim)(hidden, x)
        return hidden, nn.Dense(self.action_dim)(x)


class CommunicationEncoder(nn.Module):
    embed_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, obs):
        x = nn.relu(nn.Dense(self.hidden_dim)(obs))
        mu, log_sigma = jnp.split(nn.Dense(self.embed_dim * 2)(x), 2, -1)
        return mu, jax.nn.softplus(log_sigma) + 1e-4


class QMixer(nn.Module):
    num_agents: int
    embed_dim: int
    hidden_dim: int

    @nn.compact
    def __call__(self, agent_qs, state):
        state_dim = state.shape[-1]
        w1 = jnp.abs(nn.Dense(self.num_agents * self.embed_dim)(state))
        w1 = w1.reshape(self.num_agents, self.embed_dim)
        b1 = nn.Dense(self.embed_dim)(state)
        hidden = nn.elu(jnp.matmul(agent_qs[None, :], w1) + b1)
        w2 = jnp.abs(nn.Dense(self.embed_dim)(state)).reshape(self.embed_dim, 1)
        value = nn.Dense(self.embed_dim)(
            jnp.tanh(nn.Dense(self.hidden_dim)(state))
        )
        value = nn.Dense(1)(value).squeeze(-1)
        del state_dim
        return jnp.matmul(hidden, w2).squeeze() + value


def _stack(mapping: dict[str, jax.Array], agents: list[str]) -> jax.Array:
    return jnp.stack([mapping[agent] for agent in agents], axis=1)


def _global_state(mapping: dict[str, jax.Array]) -> jax.Array:
    return mapping["__all__"]


def make_train(config: dict[str, Any]):
    num_envs = int(config["NUM_ENVS"])
    env = SMAXLogWrapper(
        make("SMAX", scenario=map_name_to_scenario(config["MAP_NAME"]))
    )
    wrapped_env = CTRolloutManager(env, batch_size=num_envs)
    agents = env.agents
    num_agents = env.num_agents
    obs_dim = wrapped_env.obs_size
    action_dim = wrapped_env.max_action_space
    state_dim = int(env.state_size)
    hidden_dim = int(config["HIDDEN_DIM"])
    comm_dim = int(config["COMM_EMBED_DIM"])
    agent_net = AgentQNetwork(action_dim, hidden_dim)
    comm_net = CommunicationEncoder(comm_dim, hidden_dim)
    mixer = QMixer(num_agents, int(config["MIXER_EMBED_DIM"]), hidden_dim)
    key = jax.random.PRNGKey(int(config["SEED"]))
    key, k1, k2, k3 = jax.random.split(key, 4)
    agent_params = agent_net.init(
        k1, jnp.zeros((hidden_dim,)), jnp.zeros((obs_dim,)), jnp.zeros((comm_dim,))
    )
    comm_params = comm_net.init(k2, jnp.zeros((obs_dim,)))
    mixer_params = mixer.init(k3, jnp.zeros((num_agents,)), jnp.zeros((state_dim,)))
    params = {"agent": agent_params, "comm": comm_params, "mixer": mixer_params}
    tx = optax.chain(
        optax.clip_by_global_norm(float(config["MAX_GRAD_NORM"])),
        optax.radam(float(config["LR"])),
    )
    state = TrainState.create(apply_fn=lambda *_: None, params=params, tx=tx)
    target_params = params
    checkpoint_path = str(config.get("CHECKPOINT_PATH", ""))
    if checkpoint_path and Path(checkpoint_path).exists():
        with open(checkpoint_path, "rb") as checkpoint:
            restored = serialization.from_bytes(
                {"state": state, "target": target_params}, checkpoint.read()
            )
        state, target_params = restored["state"], restored["target"]

    def forward_single(network_params, hidden, obs, sample_key, sample):
        mu, sigma = comm_net.apply(network_params["comm"], obs)
        noise = jax.random.normal(sample_key, mu.shape)
        message = jnp.where(sample, mu + sigma * noise, mu)
        message = message - message.mean(axis=0, keepdims=True)
        hidden, q_values = jax.vmap(agent_net.apply, in_axes=(None, 0, 0, 0))(
            network_params["agent"],
            hidden,
            obs,
            jnp.broadcast_to(message, (num_agents, comm_dim)),
        )
        return hidden, q_values, mu, sigma, message

    def forward_batch(network_params, hidden, obs, keys, sample):
        return jax.vmap(
            forward_single, in_axes=(None, 0, 0, 0, None)
        )(network_params, hidden, obs, keys, sample)

    def train(rng: jax.Array):
        nonlocal state, target_params
        total_steps = int(config["TOTAL_TIMESTEPS"])
        rollout_steps = int(config["NUM_STEPS"])
        num_updates = max(
            1,
            (total_steps + rollout_steps * num_envs - 1)
            // (rollout_steps * num_envs),
        )
        epsilon_schedule = optax.linear_schedule(
            float(config["EPS_START"]),
            float(config["EPS_FINISH"]),
            max(1, int(config["EPS_DECAY"])),
        )

        def rollout_step(carry, _):
            obs, env_state, hidden, rng = carry
            rng, comm_keys, action_key, step_key = jax.random.split(rng, 4)
            comm_keys = jax.random.split(comm_keys, num_envs)
            next_hidden, q_values, mu, sigma, messages = forward_batch(
                state.params, hidden, _stack(obs, agents), comm_keys, True
            )
            valid_dict = wrapped_env.get_valid_actions(env_state)
            valid = jnp.stack([valid_dict[agent] for agent in agents], axis=1)
            epsilon = epsilon_schedule(rollout_index * rollout_steps)
            greedy = jnp.argmax(jnp.where(valid, q_values, -1e10), axis=-1)
            random_keys = jax.random.split(action_key, num_envs * num_agents)
            random_keys = random_keys.reshape(num_envs, num_agents, 2)

            def random_action(keys, available):
                return jax.vmap(
                    lambda k, a: jax.random.choice(
                        k, jnp.arange(action_dim), p=a / jnp.sum(a)
                    )
                )(keys, available)

            random_actions = jax.vmap(random_action)(random_keys, valid)
            actions = jnp.where(
                jax.random.uniform(action_key, (num_envs, num_agents)) < epsilon,
                random_actions,
                greedy,
            )
            action_dict = {
                agent: actions[:, index] for index, agent in enumerate(agents)
            }
            next_obs, next_state, rewards, dones, infos = wrapped_env.batch_step(
                step_key, env_state, action_dict
            )
            next_valid_dict = wrapped_env.get_valid_actions(next_state)
            next_valid = jnp.stack(
                [next_valid_dict[agent] for agent in agents], axis=1
            )
            reward = jnp.mean(
                jnp.stack([rewards[agent] for agent in agents], axis=1), axis=1
            )
            done = dones["__all__"].astype(jnp.float32)
            next_hidden = jnp.where(
                done[:, None, None].astype(bool),
                jnp.zeros_like(next_hidden),
                next_hidden,
            )
            transition = (
                _stack(obs, agents),
                _global_state(obs),
                messages,
                actions,
                valid,
                next_valid,
                reward,
                done,
                hidden,
                next_hidden,
                _stack(next_obs, agents),
                _global_state(next_obs),
                infos["returned_won_episode"],
                infos["returned_episode"],
                infos["returned_episode_returns"],
                infos["returned_episode_lengths"],
            )
            return (next_obs, next_state, next_hidden, rng), transition

        @jax.jit
        def update_fn(network_params, opt_state, target_network_params, batch, rng):
            (
                obs,
                states,
                messages,
                actions,
                valid,
                next_valid,
                rewards,
                dones,
                hidden,
                next_hidden,
                next_obs,
                next_states,
                returned_won_episode,
                returned_episode,
                returned_episode_returns,
                returned_episode_lengths,
            ) = batch

            def loss_fn(network_params):
                keys = jax.random.split(rng, rollout_steps * num_envs).reshape(
                    rollout_steps, num_envs, 2
                )
                _, q_values, mu, sigma, _ = forward_batch(
                    network_params, hidden.reshape(-1, num_agents, hidden_dim),
                    obs.reshape(-1, num_agents, obs_dim), keys.reshape(-1, 2), False
                )
                _, next_q, _, _, _ = forward_batch(
                    target_network_params,
                    next_hidden.reshape(-1, num_agents, hidden_dim),
                    next_obs.reshape(-1, num_agents, obs_dim),
                    keys.reshape(-1, 2),
                    False,
                )
                q_values = q_values.reshape(rollout_steps, num_envs, num_agents, action_dim)
                next_q = next_q.reshape(rollout_steps, num_envs, num_agents, action_dim)
                chosen_q = jnp.take_along_axis(q_values, actions[..., None], -1).squeeze(-1)
                next_max = jnp.max(jnp.where(next_valid, next_q, -1e10), axis=-1)
                mix = jax.vmap(jax.vmap(mixer.apply, in_axes=(None, 0, 0)), in_axes=(None, 0, 0))
                q_total = mix(network_params["mixer"], chosen_q, states)
                target_total = mix(target_network_params["mixer"], next_max, next_states)
                td = rewards + float(config["GAMMA"]) * (1.0 - dones) * target_total - q_total
                mu = mu.reshape(rollout_steps, num_envs, num_agents, comm_dim)
                sigma = sigma.reshape(rollout_steps, num_envs, num_agents, comm_dim)
                kl = 0.5 * (mu**2 + sigma**2 - jnp.log(sigma**2) - 1.0).mean()
                entropy = (0.5 * jnp.log(2.0 * jnp.pi * jnp.e * sigma**2)).mean()
                comm_rate = jax.nn.sigmoid(jnp.linalg.norm(mu, axis=-1)).mean()
                loss = (
                    jnp.square(td).mean()
                    + float(config["COMM_BETA"]) * kl
                    - float(config["COMM_ENTROPY_BETA"]) * entropy
                )
                return loss, (td, q_total, target_total, kl, entropy, comm_rate)

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(network_params)
            updates, opt_state = tx.update(grads, opt_state, network_params)
            network_params = optax.apply_updates(network_params, updates)
            return network_params, opt_state, (loss, aux)

        def _run_eval(key):
            """Frozen (epsilon=0) greedy evaluation, matching the dict-family baselines.

            Uses the same CTRolloutManager interface as training but with a dedicated
            batch of ``TEST_NUM_ENVS`` environments and deterministic (argmax, no
            communication noise) action selection. Returns finished-episode stats.
            """
            n_env = int(config.get("TEST_NUM_ENVS", 128))
            n_steps = int(config.get("TEST_NUM_STEPS", 128))
            eval_env = CTRolloutManager(env, batch_size=n_env)
            obs, eval_env_state = eval_env.batch_reset(key)
            hidden = jnp.zeros((n_env, num_agents, hidden_dim))

            def _eval_step(carry, _):
                obs, env_st, hidden, rng = carry
                rng, comm_key, step_key = jax.random.split(rng, 3)
                comm_keys = jax.random.split(comm_key, n_env)
                _, q_values, _, _, _ = forward_batch(
                    state.params, hidden, _stack(obs, agents), comm_keys, False
                )
                valid_dict = eval_env.get_valid_actions(env_st)
                valid = jnp.stack([valid_dict[agent] for agent in agents], axis=1)
                greedy = jnp.argmax(jnp.where(valid, q_values, -1e10), axis=-1)
                action_dict = {agent: greedy[:, i] for i, agent in enumerate(agents)}
                next_obs, next_state, _, dones, infos = eval_env.batch_step(
                    step_key, env_st, action_dict
                )
                done = dones["__all__"].astype(jnp.bool_)
                next_hidden = jnp.where(
                    done[:, None, None], jnp.zeros_like(hidden), hidden
                )
                return (next_obs, next_state, next_hidden, rng), infos

            _, infos = jax.lax.scan(
                _eval_step,
                (obs, eval_env_state, hidden, jax.random.split(key, 1)[0]),
                None,
                n_steps,
            )
            stats = finished_episode_stats(infos, num_agents)
            return {k: v for k, v in stats.items()}

        metrics_history = []
        start_time = time.time()
        obs, env_state = wrapped_env.batch_reset(rng)
        hidden = jnp.zeros((num_envs, num_agents, hidden_dim))
        for update in range(num_updates):
            rollout_index = update
            rng, rollout_key = jax.random.split(rng)
            (obs, env_state, hidden, rng), batch = jax.lax.scan(
                rollout_step,
                (obs, env_state, hidden, rollout_key),
                None,
                rollout_steps,
            )
            state_params, opt_state, (loss, aux) = update_fn(
                state.params, state.opt_state, target_params, batch, rng
            )
            state = state.replace(params=state_params, opt_state=opt_state)
            if (update + 1) % int(config["TARGET_UPDATE_INTERVAL"]) == 0:
                target_params = state.params
            td, q_taken, target, kl, entropy, comm_rate = aux
            timesteps = min((update + 1) * rollout_steps * num_envs, total_steps)

            eval_interval = max(
                1, int(config.get("TEST_INTERVAL", 0.05) * num_updates)
            )
            test_metrics = {}
            if int(config.get("TEST_DURING_TRAINING", False)) and (update + 1) % eval_interval == 0:
                rng, eval_key = jax.random.split(rng)
                test_metrics = _run_eval(eval_key)

            metrics = {                "loss/total": float(loss),
                "loss/td_error_abs": float(jnp.abs(td).mean()),
                "communication/kl": float(kl),
                "communication/entropy": float(entropy),
                "communication/rate": float(comm_rate),
                "communication_rate": float(comm_rate),
                "entropy": float(entropy),
                "q/q_taken_mean": float(q_taken.mean()),
                "q/target_mean": float(target.mean()),
                "rollout/reward_mean": float(batch[6].mean()),
                "rollout/reward_std": float(batch[6].std()),
                "rollout/done_rate": float(batch[7].mean()),
                "returned_won_episode": float(np.asarray(returned_won_episode)[-1].mean()),
                "returned_episode": float(np.asarray(returned_episode)[-1].mean()),
                "returned_episode_returns": float(
                    np.asarray(returned_episode_returns)[-1].mean()
                ),
                "returned_episode_lengths": float(
                    np.asarray(returned_episode_lengths)[-1].mean()
                ),
                "exploration/epsilon": float(epsilon_schedule(update * rollout_steps)),
                "training/update": update + 1,
                "training/timesteps": timesteps,
                "training/updates_per_second": (update + 1) / max(time.time() - start_time, 1e-6),
            }
            metrics.update({"test_" + k: v for k, v in test_metrics.items()})
            log_metrics(metrics, step=timesteps)
            metrics_history.append(metrics)
            print(
                f"NDQ update {update + 1}/{num_updates} "
                f"timesteps={timesteps}/{total_steps} "
                f"reward={metrics['rollout/reward_mean']:.4f} "
                f"return={metrics['returned_episode_returns']:.4f} "
                f"comm={metrics['communication/rate']:.3f}",
                flush=True,
            )
            if checkpoint_path and (update + 1) % int(config["CHECKPOINT_INTERVAL"]) == 0:
                Path(checkpoint_path).parent.mkdir(parents=True, exist_ok=True)
                with open(checkpoint_path, "wb") as checkpoint:
                    checkpoint.write(
                        serialization.to_bytes({"state": state, "target": target_params})
                    )
        return {"metrics": metrics_history, "runner_state": state}

    return train


def main() -> None:
    config: dict[str, Any] = {
        "MAP_NAME": "3s5z",
        "LR": 5e-4,
        "NUM_ENVS": 16,
        "NUM_STEPS": 128,
        "TOTAL_TIMESTEPS": 1_000_000,
        "GAMMA": 0.99,
        "EPS_START": 1.0,
        "EPS_FINISH": 0.05,
        "EPS_DECAY": 50000,
        "HIDDEN_DIM": 64,
        "COMM_EMBED_DIM": 3,
        "MIXER_EMBED_DIM": 32,
        "COMM_BETA": 0.001,
        "COMM_ENTROPY_BETA": 1e-6,
        "MAX_GRAD_NORM": 10.0,
        "TARGET_UPDATE_INTERVAL": 200,
        "SEED": 0,
        "PROJECT": "jaxmarl",
        "ENTITY": "",
        "WANDB_MODE": "disabled",
        "CHECKPOINT_PATH": "",
        "CHECKPOINT_INTERVAL": 1,
        "TEST_DURING_TRAINING": True,
        "TEST_INTERVAL": 0.05,
        "TEST_NUM_STEPS": 128,
        "TEST_NUM_ENVS": 512,
    }
    for argument in sys.argv[1:]:
        if "=" not in argument:
            continue
        key, value = argument.lstrip("+").split("=", 1)
        key = key[4:] if key.startswith("alg.") else key
        if key == "alg":
            continue
        if key in {
            "SEED", "NUM_ENVS", "NUM_STEPS", "TOTAL_TIMESTEPS", "EPS_DECAY",
            "HIDDEN_DIM", "COMM_EMBED_DIM", "MIXER_EMBED_DIM",
            "TARGET_UPDATE_INTERVAL", "CHECKPOINT_INTERVAL",
        }:
            config[key] = int(value)
        elif key in {
            "LR", "GAMMA", "EPS_START", "EPS_FINISH", "COMM_BETA",
            "COMM_ENTROPY_BETA", "MAX_GRAD_NORM",
        }:
            config[key] = float(value)
        elif key in {"MAP_NAME", "PROJECT", "ENTITY", "WANDB_MODE", "CHECKPOINT_PATH"}:
            config[key] = value
    run = wandb.init(
        entity=config["ENTITY"],
        project=config["PROJECT"],
        mode=config["WANDB_MODE"],
        tags=["NDQ", "SMAX", config["MAP_NAME"]],
        group=os.environ.get("WANDB_RUN_GROUP") or None,
        name=os.environ.get("WANDB_NAME") or None,
        config=config,
    )
    print(
        f"Starting JAX NDQ: project={run.project} run={run.name} "
        f"budget={config['TOTAL_TIMESTEPS']} rollout={config['NUM_STEPS']} "
        f"parallel_envs={config['NUM_ENVS']}",
        flush=True,
    )
    make_train(config)(jax.random.PRNGKey(config["SEED"]))
    run.finish()


if __name__ == "__main__":
    main()
