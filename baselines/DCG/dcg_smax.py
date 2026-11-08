"""Deep Coordination Graphs (DCG) implemented in 100% JAX and Flax for SMAX.

Based on:
- 'Deep Coordination Graphs' (Boehmer et al., ICML 2020)
- Single-file baseline structure for JaxMARL
"""

from __future__ import annotations

import functools
import os
import time
from typing import Any, NamedTuple

import chex
import distrax
import flashbax as fbx
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
# DCG Networks (Flax Linen)
# =============================================================================

class DCGNet(nn.Module):
    action_dim: int
    n_agents: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        # obs: (batch, n_agents, obs_dim)
        batch_size = obs.shape[0]

        # 1. Feature extractor per agent
        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
        h = nn.relu(h)  # (batch, n_agents, hidden_dim)

        # 2. Unary utilities f_i
        f_i = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(h)

        # 3. Pairwise payoffs f_ij for all undirected edges (i < j)
        edges_from = []
        edges_to = []
        for i in range(self.n_agents):
            for j in range(i + 1, self.n_agents):
                edges_from.append(i)
                edges_to.append(j)

        edges_from = jnp.array(edges_from, dtype=jnp.int32)
        edges_to = jnp.array(edges_to, dtype=jnp.int32)

        h_from = h[:, edges_from, :]
        h_to = h[:, edges_to, :]
        h_pairs = jnp.concatenate([h_from, h_to], axis=-1)  # (batch, n_edges, 2*hidden_dim)

        payoff_raw = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h_pairs)
        payoff_raw = nn.relu(payoff_raw)
        payoff_raw = nn.Dense(self.action_dim * self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(payoff_raw)
        f_ij = payoff_raw.reshape((batch_size, len(edges_from), self.action_dim, self.action_dim))

        # Symmetrize payoffs
        f_ij = 0.5 * (f_ij + jnp.swapaxes(f_ij, -2, -1))

        return f_i, f_ij


def message_passing(
    f_i: jax.Array,
    f_ij: jax.Array,
    edges_from: jax.Array,
    edges_to: jax.Array,
    avail_actions: jax.Array,
    n_agents: int,
    n_actions: int,
    iterations: int = 8,
) -> tuple[jax.Array, jax.Array]:
    batch_size = f_i.shape[0]
    n_edges = edges_from.shape[0]

    u = f_i / n_agents
    u = jnp.where(avail_actions == 0, -1e10, u)
    e = f_ij / n_edges

    init_msg = jnp.zeros((2, batch_size, n_edges, n_actions), dtype=jnp.float32)

    def _scatter_add_msgs(msg_to, msg_from):
        tot = jnp.zeros((batch_size, n_agents, n_actions), dtype=jnp.float32)
        tot = tot.at[:, edges_to, :].add(msg_to)
        tot = tot.at[:, edges_from, :].add(msg_from)
        return tot

    def _body_fn(step, msg):
        msg_to, msg_from = msg[0], msg[1]
        tot_msg = _scatter_add_msgs(msg_to, msg_from)

        sender0_util = u[:, edges_from, :] + (tot_msg[:, edges_from, :] - msg_from)
        joint0 = sender0_util[:, :, :, None] + e
        new_msg_to = jnp.max(joint0, axis=-2)

        sender1_util = u[:, edges_to, :] + (tot_msg[:, edges_to, :] - msg_to)
        e_transposed = jnp.swapaxes(e, -2, -1)
        joint1 = sender1_util[:, :, :, None] + e_transposed
        new_msg_from = jnp.max(joint1, axis=-2)

        new_msg_to = new_msg_to - jnp.mean(new_msg_to, axis=-1, keepdims=True)
        new_msg_from = new_msg_from - jnp.mean(new_msg_from, axis=-1, keepdims=True)
        return jnp.stack([new_msg_to, new_msg_from], axis=0)

    final_msg = jax.lax.fori_loop(0, iterations, _body_fn, init_msg)
    tot_msg = _scatter_add_msgs(final_msg[0], final_msg[1])
    agent_utils = u + tot_msg
    greedy_actions = jnp.argmax(agent_utils, axis=-1)
    return greedy_actions, agent_utils


def compute_q_tot(
    f_i: jax.Array,
    f_ij: jax.Array,
    actions: jax.Array,
    edges_from: jax.Array,
    edges_to: jax.Array,
) -> jax.Array:
    batch_size, n_agents = actions.shape
    n_edges = edges_from.shape[0]

    chosen_f_i = jnp.take_along_axis(f_i, actions[:, :, None], axis=-1).squeeze(-1)
    val_unary = jnp.mean(chosen_f_i, axis=-1)

    a_from = jnp.take_along_axis(actions, edges_from[None, :].repeat(batch_size, axis=0), axis=1)
    a_to = jnp.take_along_axis(actions, edges_to[None, :].repeat(batch_size, axis=0), axis=1)

    b_idx = jnp.arange(batch_size)[:, None]
    e_idx = jnp.arange(n_edges)[None, :]
    chosen_f_ij = f_ij[b_idx, e_idx, a_from, a_to]
    val_pairwise = jnp.mean(chosen_f_ij, axis=-1)

    return val_unary + val_pairwise


# =============================================================================
# Custom Train State & Transitions
# =============================================================================

class CustomTrainState(TrainState):
    target_params: Any
    timesteps: int = 0
    n_updates: int = 0


@chex.dataclass(frozen=True)
class Transition:
    obs: jax.Array
    actions: jax.Array
    avail_actions: jax.Array
    rewards: jax.Array
    dones: jax.Array
    next_obs: jax.Array
    next_avail: jax.Array


def make_train(config: dict):
    scenario = map_name_to_scenario(config["MAP_NAME"])
    env = HeuristicEnemySMAX(scenario=scenario, **config["ENV_KWARGS"])
    env = SMAXLogWrapper(env)

    num_agents = env.num_agents
    action_dim = env.action_space(env.agents[0]).n
    obs_dim = env.observation_space(env.agents[0]).shape[0]

    # Precompute edge indices
    edges_from = []
    edges_to = []
    for i in range(num_agents):
        for j in range(i + 1, num_agents):
            edges_from.append(i)
            edges_to.append(j)
    edges_from = jnp.array(edges_from, dtype=jnp.int32)
    edges_to = jnp.array(edges_to, dtype=jnp.int32)

    config["NUM_UPDATES"] = int(
        (config["TOTAL_TIMESTEPS"] + config["NUM_STEPS"] * config["NUM_ENVS"] - 1)
        // (config["NUM_STEPS"] * config["NUM_ENVS"])
    )

    eps_scheduler = optax.linear_schedule(
        init_value=config["EPS_START"],
        end_value=config["EPS_FINISH"],
        transition_steps=int(config["EPS_DECAY"] * config["NUM_UPDATES"]),
    )

    def train(rng: jax.Array):
        # 1. Initialize Network
        model = DCGNet(
            action_dim=action_dim,
            n_agents=num_agents,
            hidden_dim=config["HIDDEN_DIM"],
        )

        rng, rng_init = jax.random.split(rng)
        dummy_obs = jnp.zeros((config["NUM_ENVS"], num_agents, obs_dim))
        params = model.init(rng_init, dummy_obs)

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=config["LR"], eps=1e-5),
        )

        train_state = CustomTrainState.create(
            apply_fn=model.apply,
            params=params,
            target_params=params,
            tx=tx,
        )

        # 2. Initialize Buffer (Flashbax)
        dummy_trans = Transition(
            obs=jnp.zeros((num_agents, obs_dim)),
            actions=jnp.zeros((num_agents,), dtype=jnp.int32),
            avail_actions=jnp.ones((num_agents, action_dim)),
            rewards=jnp.zeros((num_agents,)),
            dones=jnp.zeros((), dtype=jnp.bool_),
            next_obs=jnp.zeros((num_agents, obs_dim)),
            next_avail=jnp.ones((num_agents, action_dim)),
        )

        buffer = fbx.make_flat_buffer(
            max_length=int(config["BUFFER_SIZE"]),
            min_length=int(config["BUFFER_BATCH_SIZE"]),
            sample_batch_size=int(config["BUFFER_BATCH_SIZE"]),
            add_sequences=False,
            add_batch_size=int(config["NUM_ENVS"] * config["NUM_STEPS"]),
        )
        buffer_state = buffer.init(dummy_trans)

        # 3. Reset Environments
        rng, rng_reset = jax.random.split(rng)
        reset_rngs = jax.random.split(rng_reset, config["NUM_ENVS"])
        obs_dict, env_state = jax.vmap(env.reset)(reset_rngs)

        def dict_to_agent_array(d, agent_list):
            return jnp.stack([d[agent] for agent in agent_list], axis=1)

        last_obs = dict_to_agent_array(obs_dict, env.agents)
        last_done = jnp.zeros(config["NUM_ENVS"], dtype=jnp.bool_)

        # 4. Training Step
        def _update_step(runner_state, update_idx):
            train_state, buffer_state, env_state, last_obs, last_done, rng = runner_state
            eps = eps_scheduler(train_state.n_updates)

            # --- COLLECT ROLLOUT ---
            def _env_step(carry, _):
                env_state, last_obs, last_done, rng = carry
                rng, rng_act, rng_eps, rng_rand, rng_step = jax.random.split(rng, 5)

                avail_actions_dict = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = dict_to_agent_array(avail_actions_dict, env.agents).astype(jnp.float32)

                f_i, f_ij = model.apply(train_state.params, last_obs)
                greedy_act, _ = message_passing(
                    f_i, f_ij, edges_from, edges_to, avail_actions,
                    num_agents, action_dim, iterations=config["MSG_ITERATIONS"]
                )

                # Epsilon-greedy action selection
                rand_logits = jnp.where(avail_actions > 0, 0.0, -1e10)
                rand_act = distrax.Categorical(logits=rand_logits).sample(seed=rng_rand)

                choose_rand = jax.random.uniform(rng_eps, greedy_act.shape) < eps
                actions = jnp.where(choose_rand, rand_act, greedy_act)

                env_act = {agent: actions[:, i] for i, agent in enumerate(env.agents)}
                step_rngs = jax.random.split(rng_step, config["NUM_ENVS"])
                next_obs_dict, env_state, rewards_dict, dones_dict, infos = jax.vmap(env.step)(
                    step_rngs, env_state, env_act
                )

                next_obs = dict_to_agent_array(next_obs_dict, env.agents)
                rewards = dict_to_agent_array(rewards_dict, env.agents)
                global_done = dones_dict["__all__"]

                next_avail_dict = jax.vmap(env.get_avail_actions)(env_state.env_state)
                next_avail = dict_to_agent_array(next_avail_dict, env.agents).astype(jnp.float32)

                trans = Transition(
                    obs=last_obs,
                    actions=actions,
                    avail_actions=avail_actions,
                    rewards=rewards,
                    dones=global_done,
                    next_obs=next_obs,
                    next_avail=next_avail,
                )

                carry = (env_state, next_obs, global_done, rng)
                return carry, (trans, infos)

            carry = (env_state, last_obs, last_done, rng)
            carry, (traj_trans, infos) = jax.lax.scan(_env_step, carry, None, config["NUM_STEPS"])
            env_state, last_obs, last_done, rng = carry

            # Flatten across (NUM_STEPS, NUM_ENVS) and push to buffer
            flat_trans = jax.tree.map(
                lambda x: x.reshape((config["NUM_STEPS"] * config["NUM_ENVS"],) + x.shape[2:]),
                traj_trans,
            )
            buffer_state = buffer.add(buffer_state, flat_trans)

            # --- Q-LEARNING UPDATE (EPOCHS) ---
            def _epoch_step(carry, _):
                train_state, buffer_state, rng = carry
                rng, rng_sample = jax.random.split(rng)

                batch = buffer.sample(buffer_state, rng_sample).experience.first

                def _loss_fn(params):
                    f_i, f_ij = model.apply(params, batch.obs)
                    q_curr = compute_q_tot(f_i, f_ij, batch.actions, edges_from, edges_to)

                    # Double Q-learning target: action selected via online model
                    next_f_i, next_f_ij = model.apply(params, batch.next_obs)
                    greedy_next_act, _ = message_passing(
                        next_f_i, next_f_ij, edges_from, edges_to, batch.next_avail,
                        num_agents, action_dim, iterations=config["MSG_ITERATIONS"]
                    )

                    # Q-value computed on target network
                    target_f_i, target_f_ij = model.apply(train_state.target_params, batch.next_obs)
                    q_target = compute_q_tot(target_f_i, target_f_ij, greedy_next_act, edges_from, edges_to)

                    global_reward = jnp.mean(batch.rewards, axis=-1)
                    target_y = global_reward + config["GAMMA"] * (1.0 - batch.dones) * q_target
                    td_loss = jnp.mean(jnp.square(q_curr - jax.lax.stop_gradient(target_y)))
                    return td_loss

                loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                return (train_state, buffer_state, rng), loss

            (train_state, buffer_state, rng), losses = jax.lax.scan(
                _epoch_step,
                (train_state, buffer_state, rng),
                None,
                config["NUM_EPOCHS"],
            )

            # Periodic Target Update
            new_target = jax.tree.map(
                lambda p, tp: jnp.where((update_idx % config["TARGET_UPDATE_INTERVAL"]) == 0, p, tp),
                train_state.params,
                train_state.target_params,
            )
            train_state = train_state.replace(
                target_params=new_target,
                n_updates=train_state.n_updates + 1,
            )

            metrics = {
                "loss": jnp.mean(losses),
                "won_rate": jnp.mean(infos["returned_won_episode"]),
                "returns": jnp.mean(infos["returned_episode_returns"]),
            }

            runner_state = (train_state, buffer_state, env_state, last_obs, last_done, rng)
            return runner_state, metrics

        runner_state = (train_state, buffer_state, env_state, last_obs, last_done, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


@hydra.main(version_base=None, config_path="config", config_name="dcg_smax")
def main(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config.get("ENTITY", None),
        project=config["WANDB_PROJECT"],
        tags=["DCG", "SMAX", config["MAP_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config.get("SEED", 42))
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    print("DCG training on SMAX 3s5z completed successfully!")


if __name__ == "__main__":
    main()
