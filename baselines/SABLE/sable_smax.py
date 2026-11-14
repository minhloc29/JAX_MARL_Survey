"""SABLE (Scalable Agent-By-Agent LEarning) in 100% JAX/Flax for SMAX.

Based on:
- 'Scalable Agent-By-Agent LEarning' (Mava/InstaDeep, 2024)
- RetNet: 'Retentive Network: A Successor to Transformer for Large Language Models' (Sun et al., 2023)
- Reference: temp/Mava/mava/networks/retention.py & ff_sable.py
"""

from __future__ import annotations

import functools
import os
import time
from typing import NamedTuple, Sequence

import distrax
import flax.linen as nn
import hydra
import jax
import jax.numpy as jnp
import numpy as np
import optax
from flax.linen.initializers import constant, orthogonal
from flax.training.train_state import TrainState
from omegaconf import OmegaConf

import wandb
from jaxmarl import make
from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import SMAXLogWrapper


# =============================================================================
# SABLE / RetNet Architecture (100% Flax Linen)
# =============================================================================

class MultiScaleRetention(nn.Module):
    embed_dim: int
    n_heads: int
    masked: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        batch_size, seq_len, _ = x.shape
        head_dim = self.embed_dim // self.n_heads

        q = nn.Dense(self.embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        k = nn.Dense(self.embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        v = nn.Dense(self.embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)

        # Reshape to (batch, n_heads, seq_len, head_dim)
        q = q.reshape(batch_size, seq_len, self.n_heads, head_dim).swapaxes(1, 2)
        k = k.reshape(batch_size, seq_len, self.n_heads, head_dim).swapaxes(1, 2)
        v = v.reshape(batch_size, seq_len, self.n_heads, head_dim).swapaxes(1, 2)

        # Multi-scale decay factors: gamma_h = 1 - 2^(-5 - h)
        decay_rates = 1.0 - 2.0 ** (-5.0 - jnp.arange(self.n_heads, dtype=jnp.float32))
        
        i_idx = jnp.arange(seq_len)[:, None]
        j_idx = jnp.arange(seq_len)[None, :]
        dist = i_idx - j_idx  # (seq_len, seq_len)

        decay_matrix = decay_rates[:, None, None] ** dist[None, :, :]
        if self.masked:
            causal_mask = dist >= 0
            decay_matrix = jnp.where(causal_mask[None, :, :], decay_matrix, 0.0)

        # Retention score
        qk = jnp.matmul(q, jnp.swapaxes(k, -1, -2)) / np.sqrt(head_dim)
        ret_weights = qk * decay_matrix[None, :, :, :]
        ret = jnp.matmul(ret_weights, v)

        # Merge heads
        ret = ret.swapaxes(1, 2).reshape(batch_size, seq_len, self.embed_dim)
        out = nn.Dense(self.embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(ret)
        return out


class SableBlock(nn.Module):
    embed_dim: int
    n_heads: int
    masked: bool = False

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        # Retention branch
        norm1 = nn.LayerNorm()(x)
        ret = MultiScaleRetention(
            embed_dim=self.embed_dim,
            n_heads=self.n_heads,
            masked=self.masked,
        )(norm1)
        x = x + ret

        # Feed-forward branch
        norm2 = nn.LayerNorm()(x)
        ff = nn.Dense(self.embed_dim * 2, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(norm2)
        ff = nn.gelu(ff)
        ff = nn.Dense(self.embed_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(ff)
        x = x + ff
        return x


class SableModel(nn.Module):
    action_dim: int
    n_agents: int
    n_embd: int = 64
    n_heads: int = 2
    n_blocks: int = 1

    @nn.compact
    def __call__(
        self,
        obs: jax.Array,
        avail_actions: jax.Array,
    ) -> tuple[distrax.Categorical, jax.Array]:
        batch_size, n_agents, _ = obs.shape

        # 1. Observation projection
        x = nn.Dense(self.n_embd, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)

        # 2. Sable RetNet Blocks
        for _ in range(self.n_blocks):
            x = SableBlock(embed_dim=self.n_embd, n_heads=self.n_heads, masked=False)(x)

        # 3. Policy Head
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        large_neg = -1e10
        masked_logits = jnp.where(avail_actions > 0.5, logits, large_neg)
        pi = distrax.Categorical(logits=masked_logits)

        # 4. Centralized Critic Head
        flat_obs = obs.reshape(batch_size, -1)
        v = nn.Dense(self.n_embd, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(flat_obs)
        v = nn.gelu(v)
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v)
        value = jnp.tile(v, (1, n_agents))

        return pi, value


# =============================================================================
# Training Pipeline
# =============================================================================

class Transition(NamedTuple):
    done: jnp.ndarray          # (n_envs,)
    action: jnp.ndarray        # (n_envs, n_agents)
    value: jnp.ndarray         # (n_envs, n_agents)
    reward: jnp.ndarray        # (n_envs, n_agents)
    log_prob: jnp.ndarray      # (n_envs, n_agents)
    obs: jnp.ndarray           # (n_envs, n_agents, obs_dim)
    info: dict
    avail_actions: jnp.ndarray # (n_envs, n_agents, act_dim)


def make_train(config: dict):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    env = SMAXLogWrapper(env)

    num_agents = env.num_agents
    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]

    config["NUM_UPDATES"] = int(
        (config["TOTAL_TIMESTEPS"] + config["NUM_STEPS"] * config["NUM_ENVS"] - 1)
        // (config["NUM_STEPS"] * config["NUM_ENVS"])
    )
    config["MINIBATCH_SIZE"] = (
        config["NUM_ENVS"] * config["NUM_STEPS"] // config["NUM_MINIBATCHES"]
    )

    def linear_schedule(count):
        frac = 1.0 - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"])) / config["NUM_UPDATES"]
        return config["LR"] * jnp.maximum(frac, 0.0)

    def train(rng: jax.Array):
        # 1. Initialize Network
        model = SableModel(
            action_dim=action_dim,
            n_agents=num_agents,
            n_embd=config["N_EMBD"],
            n_heads=config["N_HEADS"],
            n_blocks=config["N_BLOCKS"],
        )

        rng, rng_init = jax.random.split(rng)
        dummy_obs = jnp.zeros((config["NUM_ENVS"], num_agents, obs_dim))
        dummy_avail = jnp.ones((config["NUM_ENVS"], num_agents, action_dim))

        params = model.init(rng_init, obs=dummy_obs, avail_actions=dummy_avail)

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(
                learning_rate=linear_schedule if config["ANNEAL_LR"] else config["LR"],
                eps=1e-5,
            ),
        )
        train_state = TrainState.create(apply_fn=model.apply, params=params, tx=tx)

        # 2. Reset Parallel Environments
        rng, rng_reset = jax.random.split(rng)
        reset_rngs = jax.random.split(rng_reset, config["NUM_ENVS"])
        obs_dict, env_state = jax.vmap(env.reset)(reset_rngs)

        def dict_to_agent_array(d, agent_list):
            return jnp.stack([d[agent] for agent in agent_list], axis=1)

        last_obs = dict_to_agent_array(obs_dict, env.agents)
        last_done = jnp.zeros(config["NUM_ENVS"], dtype=jnp.bool_)

        def _log_update(metrics, update_index):
            if wandb.run is not None:
                wandb.log(
                    {key: float(value) for key, value in metrics.items()},
                    step=int(update_index)
                    * int(config["NUM_STEPS"])
                    * int(config["NUM_ENVS"]),
                )

        # 3. Main Update Loop
        def _update_step(runner_state, update_index):
            train_state, env_state, last_obs, last_done, rng = runner_state

            # --- ROLLOUT ---
            def _env_step(carry, _):
                env_state, last_obs, last_done, rng = carry
                rng, rng_avail, rng_act, rng_step = jax.random.split(rng, 4)

                avail_actions_dict = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = dict_to_agent_array(avail_actions_dict, env.agents)

                pi, value = model.apply(
                    train_state.params,
                    obs=last_obs,
                    avail_actions=avail_actions,
                )
                action = pi.sample(seed=rng_act)
                log_prob = pi.log_prob(action)

                actions_dict = {
                    agent: action[:, i] for i, agent in enumerate(env.agents)
                }

                step_rngs = jax.random.split(rng_step, config["NUM_ENVS"])
                obs_dict, env_state, rewards_dict, dones_dict, infos = jax.vmap(
                    env.step
                )(step_rngs, env_state, actions_dict)

                obs = dict_to_agent_array(obs_dict, env.agents)
                reward = dict_to_agent_array(rewards_dict, env.agents)
                done = dones_dict["__all__"]

                transition = Transition(
                    done=last_done,
                    action=action,
                    value=value,
                    reward=reward,
                    log_prob=log_prob,
                    obs=last_obs,
                    info=infos,
                    avail_actions=avail_actions,
                )

                carry = (env_state, obs, done, rng)
                return carry, transition

            runner_carry = (env_state, last_obs, last_done, rng)
            runner_carry, traj_batch = jax.lax.scan(
                _env_step, runner_carry, None, length=config["NUM_STEPS"]
            )
            env_state, last_obs, last_done, rng = runner_carry

            # --- GAE CALCULATION ---
            avail_actions_dict = jax.vmap(env.get_avail_actions)(env_state.env_state)
            avail_actions = dict_to_agent_array(avail_actions_dict, env.agents)
            _, last_val = model.apply(
                train_state.params,
                obs=last_obs,
                avail_actions=avail_actions,
            )

            def _calculate_gae(traj_batch, last_val, last_done):
                def _get_advantages(carry, transition_and_done):
                    gae, next_val, next_done = carry
                    transition, done = transition_and_done
                    delta = (
                        transition.reward
                        + config["GAMMA"] * next_val * (1.0 - next_done[:, None])
                        - transition.value
                    )
                    gae = (
                        delta
                        + config["GAMMA"]
                        * config["GAE_LAMBDA"]
                        * (1.0 - next_done[:, None])
                        * gae
                    )
                    return (gae, transition.value, done), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val, last_done),
                    (traj_batch, traj_batch.done),
                    reverse=True,
                )
                returns = advantages + traj_batch.value
                return advantages, returns

            advantages, targets = _calculate_gae(traj_batch, last_val, last_done)

            # --- PPO UPDATE ---
            def _update_epoch(update_state, _):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets = batch_info

                    def _loss_fn(params):
                        pi, value = model.apply(
                            params,
                            obs=traj_batch.obs,
                            avail_actions=traj_batch.avail_actions,
                        )
                        log_prob = pi.log_prob(traj_batch.action)

                        # Policy Loss
                        ratio = jnp.exp(log_prob - traj_batch.log_prob)
                        norm_adv = (advantages - advantages.mean()) / (
                            advantages.std() + 1e-8
                        )
                        surr1 = ratio * norm_adv
                        surr2 = (
                            jnp.clip(
                                ratio,
                                1.0 - config["CLIP_EPS"],
                                1.0 + config["CLIP_EPS"],
                            )
                            * norm_adv
                        )
                        actor_loss = -jnp.minimum(surr1, surr2).mean()

                        # Value Loss
                        v_clipped = traj_batch.value + jnp.clip(
                            value - traj_batch.value,
                            -config["CLIP_EPS"],
                            config["CLIP_EPS"],
                        )
                        v_loss1 = jnp.square(value - targets)
                        v_loss2 = jnp.square(v_clipped - targets)
                        critic_loss = 0.5 * jnp.maximum(v_loss1, v_loss2).mean()

                        # Entropy Loss
                        entropy = pi.entropy().mean()

                        total_loss = (
                            actor_loss
                            + config["VF_COEF"] * critic_loss
                            - config["ENT_COEF"] * entropy
                        )
                        return total_loss, (actor_loss, critic_loss, entropy)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss, (a_loss, c_loss, ent)), grads = grad_fn(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss, a_loss, c_loss, ent)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, rng_perm = jax.random.split(rng)

                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                perm = jax.random.permutation(rng_perm, batch_size)

                batch = jax.tree_util.tree_map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]),
                    (traj_batch, advantages, targets),
                )
                shuffled_batch = jax.tree_util.tree_map(
                    lambda x: jnp.take(x, perm, axis=0), batch
                )
                minibatches = jax.tree_util.tree_map(
                    lambda x: jnp.reshape(
                        x, (config["NUM_MINIBATCHES"], -1) + x.shape[1:]
                    ),
                    shuffled_batch,
                )

                train_state, (loss, a_loss, c_loss, ent) = jax.lax.scan(
                    _update_minibatch, train_state, minibatches
                )
                return (train_state, traj_batch, advantages, targets, rng), (
                    loss,
                    a_loss,
                    c_loss,
                    ent,
                )

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, (loss, a_loss, c_loss, ent) = jax.lax.scan(
                _update_epoch, update_state, None, length=config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # --- METRICS & LOGGING ---
            metrics = {
                "loss": loss.mean(),
                "actor_loss": a_loss.mean(),
                "critic_loss": c_loss.mean(),
                "entropy": ent.mean(),
                "returned_won_episode": jnp.mean(traj_batch.info["returned_won_episode"]),
                "returned_episode_returns": jnp.mean(
                    traj_batch.info["returned_episode_returns"]
                ),
                "returned_episode_lengths": jnp.mean(
                    traj_batch.info["returned_episode_lengths"]
                ),
                "returned_episode": jnp.mean(traj_batch.info["returned_episode"]),
            }
            jax.debug.callback(_log_update, metrics, update_index + 1)

            runner_state = (train_state, env_state, last_obs, last_done, rng)
            return runner_state, metrics

        # 4. Run All Updates via lax.scan
        runner_state = (train_state, env_state, last_obs, last_done, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


@hydra.main(version_base=None, config_path="config", config_name="sable_smax")
def main(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config.get("ENTITY", None),
        project=config["WANDB_PROJECT"],
        tags=["SABLE", "SMAX", config["MAP_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config.get("SEED", 42))
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    wandb.finish()
    print("SABLE training on SMAX 3s5z completed successfully!")


if __name__ == "__main__":
    main()
