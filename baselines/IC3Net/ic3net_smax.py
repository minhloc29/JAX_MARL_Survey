"""IC3Net (Individual Controlled Continuous Communication Network) in 100% JAX/Flax for SMAX.

Based on:
- 'Learning when to Communicate at Scale in Multiagent Cooperative and Competitive Settings' (Singh et al., ICLR 2019)
- Reference: IC3Net/comm.py
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
# IC3Net Architecture
# =============================================================================

class IC3NetCommBlock(nn.Module):
    hidden_dim: int

    @nn.compact
    def __call__(self, h: jax.Array, gate: jax.Array) -> jax.Array:
        """Gated continuous communication pass across agents.
        
        Args:
            h: agent representations of shape (batch, n_agents, hidden_dim)
            gate: gating action (0 or 1) of shape (batch, n_agents, 1)
        Returns:
            updated representations of shape (batch, n_agents, hidden_dim)
        """
        batch_size, n_agents, hidden_dim = h.shape

        # Active messages sent by talking agents: (batch, n_agents, hidden_dim)
        messages = h * gate

        # Broadcast and exclude self communication
        # all_messages: (batch, 1, n_agents, hidden_dim)
        # mask: eye(n_agents) is 0 on diagonal, 1 off diagonal
        eye_mask = 1.0 - jnp.eye(n_agents)  # (n_agents, n_agents)
        eye_mask = eye_mask[None, :, :, None]  # (1, n_agents, n_agents, 1)

        # Pairwise incoming messages: (batch, n_agents, n_agents, hidden_dim)
        peer_messages = messages[:, None, :, :] * eye_mask

        # Sum incoming messages from talking peers
        comm_sum = jnp.sum(peer_messages, axis=2)  # (batch, n_agents, hidden_dim)

        # Count talking peers for each agent
        peer_gates = gate[:, None, :, :] * eye_mask  # (batch, n_agents, n_agents, 1)
        peer_counts = jnp.sum(peer_gates, axis=2)    # (batch, n_agents, 1)

        # Average incoming communication (0 when no peers communicate)
        comm_avg = comm_sum / jnp.maximum(peer_counts, 1.0)
        c = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(comm_avg)

        # Skip connection & activation
        h_next = nn.tanh(h + c)
        return h_next


class IC3NetActorCritic(nn.Module):
    action_dim: int
    n_agents: int
    hidden_dim: int = 64
    comm_passes: int = 1

    @nn.compact
    def __call__(
        self,
        obs: jax.Array,
        avail_actions: jax.Array,
        gate: jax.Array | None = None,
    ) -> tuple[distrax.Categorical, distrax.Categorical, jax.Array]:
        """Forward pass for environment policy, gate policy, and centralized critic.
        
        Args:
            obs: (batch, n_agents, obs_dim)
            avail_actions: (batch, n_agents, action_dim)
            gate: optional gating action of shape (batch, n_agents)
        """
        batch_size, n_agents, _ = obs.shape

        # 1. Local observation encoder
        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        h = nn.tanh(h)

        # 2. Communication Gate Head (discrete binary: 0 = silent, 1 = talk)
        gate_logits = nn.Dense(2, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(h)
        gate_pi = distrax.Categorical(logits=gate_logits)

        # Use provided gate or gate mode
        if gate is None:
            gate_val = jnp.argmax(gate_logits, axis=-1, keepdims=True).astype(jnp.float32)
        else:
            gate_val = gate[..., None].astype(jnp.float32)

        # 3. Gated Communication Passes
        for _ in range(self.comm_passes):
            h = IC3NetCommBlock(hidden_dim=self.hidden_dim)(h, gate_val)

        # 4. Environment Policy Head
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(h)
        large_neg = -1e10
        masked_logits = jnp.where(avail_actions > 0.5, logits, large_neg)
        env_pi = distrax.Categorical(logits=masked_logits)

        # 5. Centralized Critic Head
        flat_obs = obs.reshape(batch_size, -1)
        v = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(flat_obs)
        v = nn.tanh(v)
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v)
        value = jnp.tile(v, (1, n_agents))

        return env_pi, gate_pi, value


# =============================================================================
# Training Pipeline
# =============================================================================

class Transition(NamedTuple):
    done: jnp.ndarray          # (n_envs,)
    action: jnp.ndarray        # (n_envs, n_agents)
    gate: jnp.ndarray          # (n_envs, n_agents)
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
        model = IC3NetActorCritic(
            action_dim=action_dim,
            n_agents=num_agents,
            hidden_dim=config["HIDDEN_DIM"],
            comm_passes=config["COMM_PASSES"],
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
                rng, rng_avail, rng_gate, rng_act, rng_step = jax.random.split(rng, 5)

                avail_actions_dict = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = dict_to_agent_array(avail_actions_dict, env.agents)

                # Gate action sampled first
                _, gate_pi, _ = model.apply(
                    train_state.params,
                    obs=last_obs,
                    avail_actions=avail_actions,
                )
                gate = gate_pi.sample(seed=rng_gate)
                gate_log_prob = gate_pi.log_prob(gate)

                # Environment action and value conditional on sampled gate
                env_pi, _, value = model.apply(
                    train_state.params,
                    obs=last_obs,
                    avail_actions=avail_actions,
                    gate=gate,
                )
                action = env_pi.sample(seed=rng_act)
                env_log_prob = env_pi.log_prob(action)

                # Joint log prob
                joint_log_prob = env_log_prob + gate_log_prob

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
                    gate=gate,
                    value=value,
                    reward=reward,
                    log_prob=joint_log_prob,
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
            _, _, last_val = model.apply(
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
                        env_pi, gate_pi, value = model.apply(
                            params,
                            obs=traj_batch.obs,
                            avail_actions=traj_batch.avail_actions,
                            gate=traj_batch.gate,
                        )
                        env_log_prob = env_pi.log_prob(traj_batch.action)
                        gate_log_prob = gate_pi.log_prob(traj_batch.gate)
                        joint_log_prob = env_log_prob + gate_log_prob

                        # Policy Loss
                        ratio = jnp.exp(joint_log_prob - traj_batch.log_prob)
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

                        # Joint Entropy Loss
                        entropy = (env_pi.entropy() + gate_pi.entropy()).mean()

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
                "comm_gate_rate": jnp.mean(traj_batch.gate),
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


@hydra.main(version_base=None, config_path="config", config_name="ic3net_smax")
def main(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config.get("ENTITY", None),
        project=config["WANDB_PROJECT"],
        tags=["IC3NET", "SMAX", config["MAP_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config.get("SEED", 42))
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    wandb.finish()
    print("IC3Net training on SMAX 3s5z completed successfully!")


if __name__ == "__main__":
    main()
