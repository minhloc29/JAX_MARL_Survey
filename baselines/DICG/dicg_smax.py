"""Deep Implicit Coordination Graphs (DICG) implemented in 100% JAX and Flax for SMAX.

Based on:
- 'Deep Implicit Coordination Graphs for Multi-Agent Reinforcement Learning' (Li et al., NeurIPS 2020)
- Single-file baseline structure for JaxMARL
"""

from __future__ import annotations

import functools
import os
import time
from typing import NamedTuple

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
# DICG Neural Modules (100% Flax Linen)
# =============================================================================

class GraphConvolutionLayer(nn.Module):
    out_features: int
    bias: bool = True

    @nn.compact
    def __call__(self, h: jax.Array, adj: jax.Array) -> jax.Array:
        # h: (batch, n_agents, in_features)
        # adj: (batch, n_agents, n_agents)
        support = nn.Dense(self.out_features, use_bias=self.bias, kernel_init=orthogonal(np.sqrt(2)))(h)
        outputs = jnp.matmul(adj, support)
        return jnp.tanh(outputs)


class AttentionLayer(nn.Module):
    dimensions: int
    attention_type: str = "general"

    @nn.compact
    def __call__(self, query: jax.Array) -> jax.Array:
        # query: (batch, n_agents, dimensions)
        if self.attention_type == "general":
            q_proj = nn.Dense(self.dimensions, use_bias=False, kernel_init=orthogonal(np.sqrt(2)))(query)
            context = jnp.swapaxes(query, -2, -1)
            attention_scores = jnp.matmul(q_proj, context)
        else:
            # dot product
            context = jnp.swapaxes(query, -2, -1)
            attention_scores = jnp.matmul(query, context)

        attention_weights = jax.nn.softmax(attention_scores, axis=-1)
        return attention_weights


class DICGBase(nn.Module):
    embedding_dim: int = 64
    n_gcn_layers: int = 2
    attention_type: str = "general"
    residual: bool = True

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        # obs: (batch, n_agents, obs_dim)
        # 1. MLP Encoder
        h0 = nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        h0 = jnp.tanh(h0)
        h0 = nn.Dense(self.embedding_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h0)
        h0 = jnp.tanh(h0)

        # 2. Dynamic Attention Adjacency Matrix
        adj = AttentionLayer(dimensions=self.embedding_dim, attention_type=self.attention_type)(h0)

        # 3. GCN Layers
        h = h0
        for _ in range(self.n_gcn_layers):
            h = GraphConvolutionLayer(out_features=self.embedding_dim)(h, adj)

        # 4. Residual Connection
        h_out = (h0 + h) if self.residual else h
        return h_out, adj


class DICGModel(nn.Module):
    action_dim: int
    n_agents: int
    embedding_dim: int = 64
    n_gcn_layers: int = 2
    attention_type: str = "general"
    residual: bool = True

    def setup(self):
        self.actor_base = DICGBase(
            embedding_dim=self.embedding_dim,
            n_gcn_layers=self.n_gcn_layers,
            attention_type=self.attention_type,
            residual=self.residual,
        )
        self.critic_base = DICGBase(
            embedding_dim=self.embedding_dim,
            n_gcn_layers=self.n_gcn_layers,
            attention_type=self.attention_type,
            residual=self.residual,
        )

        # Actor categorical output
        self.actor_head = nn.Sequential([
            nn.Dense(128, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0)),
        ])

        # Critic value output
        self.critic_head = nn.Sequential([
            nn.Dense(64, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0)),
            nn.tanh,
            nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0)),
        ])

    def get_actions(
        self, rng: jax.Array, obs: jax.Array, avail_actions: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # obs: (batch, n_agents, obs_dim)
        actor_emb, _ = self.actor_base(obs)
        logits = self.actor_head(actor_emb)
        logits = logits - (1.0 - avail_actions) * 1e10

        pi = distrax.Categorical(logits=logits)
        actions = pi.sample(seed=rng)
        log_prob = pi.log_prob(actions)

        critic_emb, _ = self.critic_base(obs)
        values = self.critic_head(critic_emb).squeeze(-1)
        return actions, log_prob, values

    def evaluate_actions(
        self, obs: jax.Array, actions: jax.Array, avail_actions: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        actor_emb, _ = self.actor_base(obs)
        logits = self.actor_head(actor_emb)
        logits = logits - (1.0 - avail_actions) * 1e10

        pi = distrax.Categorical(logits=logits)
        log_prob = pi.log_prob(actions)
        entropy = pi.entropy()

        critic_emb, _ = self.critic_base(obs)
        values = self.critic_head(critic_emb).squeeze(-1)
        return log_prob, values, entropy

    def get_values(self, obs: jax.Array) -> jax.Array:
        critic_emb, _ = self.critic_base(obs)
        return self.critic_head(critic_emb).squeeze(-1)


# =============================================================================
# Rollout & Training Structures
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
        model = DICGModel(
            action_dim=action_dim,
            n_agents=num_agents,
            embedding_dim=config["EMBEDDING_DIM"],
            n_gcn_layers=config["N_GCN_LAYERS"],
            attention_type=config["ATTENTION_TYPE"],
            residual=config["RESIDUAL"],
        )

        rng, rng_init = jax.random.split(rng)
        dummy_obs = jnp.zeros((config["NUM_ENVS"], num_agents, obs_dim))
        dummy_act = jnp.zeros((config["NUM_ENVS"], num_agents), dtype=jnp.int32)
        dummy_avail = jnp.ones((config["NUM_ENVS"], num_agents, action_dim))

        params = model.init(
            rng_init,
            obs=dummy_obs,
            actions=dummy_act,
            avail_actions=dummy_avail,
            method=model.evaluate_actions,
        )

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

        # 3. Main Update Loop
        def _update_step(runner_state, unused):
            train_state, env_state, last_obs, last_done, rng = runner_state

            # --- ROLLOUT ---
            def _env_step(carry, _):
                env_state, last_obs, last_done, rng = carry
                rng, rng_act, rng_step = jax.random.split(rng, 3)

                avail_actions_dict = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = dict_to_agent_array(avail_actions_dict, env.agents)

                actions, log_probs, values = model.apply(
                    train_state.params,
                    rng=rng_act,
                    obs=last_obs,
                    avail_actions=avail_actions,
                    method=model.get_actions,
                )

                env_act = {agent: actions[:, i] for i, agent in enumerate(env.agents)}
                step_rngs = jax.random.split(rng_step, config["NUM_ENVS"])
                next_obs_dict, env_state, rewards_dict, dones_dict, infos = jax.vmap(env.step)(
                    step_rngs, env_state, env_act
                )

                next_obs = dict_to_agent_array(next_obs_dict, env.agents)
                rewards = dict_to_agent_array(rewards_dict, env.agents)
                global_done = dones_dict["__all__"]

                transition = Transition(
                    done=global_done,
                    action=actions,
                    value=values,
                    reward=rewards,
                    log_prob=log_probs,
                    obs=last_obs,
                    info=infos,
                    avail_actions=avail_actions,
                )

                carry = (env_state, next_obs, global_done, rng)
                return carry, transition

            carry = (env_state, last_obs, last_done, rng)
            carry, traj_batch = jax.lax.scan(_env_step, carry, None, config["NUM_STEPS"])
            env_state, last_obs, last_done, rng = carry

            # --- CALCULATE GAE ---
            last_val = model.apply(train_state.params, obs=last_obs, method=model.get_values)

            def _calculate_gae(traj_batch, last_val):
                def _get_advantages(gae_and_next_value, transition):
                    gae, next_value = gae_and_next_value
                    done, value, reward = (
                        transition.done[:, None],
                        transition.value,
                        transition.reward,
                    )
                    delta = reward + config["GAMMA"] * next_value * (1.0 - done) - value
                    gae = delta + config["GAMMA"] * config["GAE_LAMBDA"] * (1.0 - done) * gae
                    return (gae, value), gae

                _, advantages = jax.lax.scan(
                    _get_advantages,
                    (jnp.zeros_like(last_val), last_val),
                    traj_batch,
                    reverse=True,
                    unroll=16,
                )
                returns = advantages + traj_batch.value
                return advantages, returns

            advantages, returns = _calculate_gae(traj_batch, last_val)

            # --- PPO UPDATE ---
            def _update_epoch(update_state, _):
                train_state, traj_batch, advantages, returns, rng = update_state
                rng, rng_perm = jax.random.split(rng)

                batch_size = config["NUM_STEPS"] * config["NUM_ENVS"]
                flat_batch = jax.tree.map(
                    lambda x: x.reshape((batch_size,) + x.shape[2:]), traj_batch
                )
                flat_adv = advantages.reshape((batch_size, num_agents))
                flat_ret = returns.reshape((batch_size, num_agents))

                flat_adv = (flat_adv - flat_adv.mean()) / (flat_adv.std() + 1e-8)

                permutation = jax.random.permutation(rng_perm, batch_size)
                batch = jax.tree.map(lambda x: x[permutation], flat_batch)
                b_adv = flat_adv[permutation]
                b_ret = flat_ret[permutation]

                def _minibatch_step(train_state, i):
                    idx = jax.lax.dynamic_slice_in_dim(
                        jnp.arange(batch_size), i * config["MINIBATCH_SIZE"], config["MINIBATCH_SIZE"]
                    )
                    mb = jax.tree.map(lambda x: x[idx], batch)
                    mb_adv = b_adv[idx]
                    mb_ret = b_ret[idx]

                    def _loss_fn(params):
                        log_prob, value, entropy = model.apply(
                            params,
                            obs=mb.obs,
                            actions=mb.action,
                            avail_actions=mb.avail_actions,
                            method=model.evaluate_actions,
                        )

                        ratio = jnp.exp(log_prob - mb.log_prob)
                        surr1 = ratio * mb_adv
                        surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * mb_adv
                        actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))

                        v_clipped = mb.value + jnp.clip(
                            value - mb.value, -config["CLIP_EPS"], config["CLIP_EPS"]
                        )
                        vf_loss1 = jnp.square(value - mb_ret)
                        vf_loss2 = jnp.square(v_clipped - mb_ret)
                        critic_loss = 0.5 * jnp.mean(jnp.maximum(vf_loss1, vf_loss2))

                        entropy_loss = jnp.mean(entropy)
                        total_loss = (
                            actor_loss
                            + config["VF_COEF"] * critic_loss
                            - config["ENT_COEF"] * entropy_loss
                        )
                        return total_loss, (actor_loss, critic_loss, entropy_loss)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss, (act_l, crit_l, ent_l)), grads = grad_fn(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss, act_l, crit_l, ent_l)

                train_state, metrics = jax.lax.scan(
                    _minibatch_step,
                    train_state,
                    jnp.arange(config["NUM_MINIBATCHES"]),
                )
                update_state = (train_state, traj_batch, advantages, returns, rng)
                return update_state, metrics

            update_state = (train_state, traj_batch, advantages, returns, rng)
            update_state, loss_metrics = jax.lax.scan(
                _update_epoch, update_state, None, config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            metrics = {
                "loss": jnp.mean(loss_metrics[0]),
                "won_rate": jnp.mean(traj_batch.info["returned_won_episode"]),
                "returns": jnp.mean(traj_batch.info["returned_episode_returns"]),
            }

            runner_state = (train_state, env_state, last_obs, last_done, rng)
            return runner_state, metrics

        runner_state = (train_state, env_state, last_obs, last_done, rng)
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, None, config["NUM_UPDATES"]
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


@hydra.main(version_base=None, config_path="config", config_name="dicg_smax")
def main(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config.get("ENTITY", None),
        project=config["WANDB_PROJECT"],
        tags=["DICG", "SMAX", config["MAP_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config.get("SEED", 42))
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    print("DICG training on SMAX 3s5z completed successfully!")


if __name__ == "__main__":
    main()
