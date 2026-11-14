"""Multi-Agent Transformer (MAT) implemented in 100% JAX and Flax for SMAX.

Based on:
- 'Multi-Agent PPO: Multi-Agent Actor-Critic as Sequence Modeling' (Meng et al., 2022)
- CleanRL / PureJaxRL single-file architecture for JaxMARL
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
# Transformer Architecture (100% Flax Linen)
# =============================================================================

class MultiHeadAttention(nn.Module):
    n_embd: int
    n_head: int
    masked: bool = False

    @nn.compact
    def __call__(self, query: jax.Array, key: jax.Array, value: jax.Array) -> jax.Array:
        batch_size, seq_len, _ = query.shape
        head_dim = self.n_embd // self.n_head

        q = nn.Dense(self.n_embd, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(query)
        k = nn.Dense(self.n_embd, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(key)
        v = nn.Dense(self.n_embd, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(value)

        # Reshape to (batch, n_head, seq_len, head_dim)
        q = q.reshape((batch_size, seq_len, self.n_head, head_dim)).swapaxes(1, 2)
        k_len = key.shape[1]
        k = k.reshape((batch_size, k_len, self.n_head, head_dim)).swapaxes(1, 2)
        v = v.reshape((batch_size, k_len, self.n_head, head_dim)).swapaxes(1, 2)

        # Scaled dot-product attention
        att = jnp.matmul(q, k.swapaxes(-2, -1)) * (1.0 / np.sqrt(head_dim))

        if self.masked:
            causal_mask = jnp.tril(jnp.ones((seq_len, k_len)))
            att = jnp.where(causal_mask == 0, -1e9, att)

        att = jax.nn.softmax(att, axis=-1)
        y = jnp.matmul(att, v)  # (batch, n_head, seq_len, head_dim)
        y = y.swapaxes(1, 2).reshape((batch_size, seq_len, self.n_embd))

        return nn.Dense(self.n_embd, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(y)


class EncodeBlock(nn.Module):
    n_embd: int
    n_head: int

    @nn.compact
    def __call__(self, x: jax.Array) -> jax.Array:
        attn_out = MultiHeadAttention(self.n_embd, self.n_head, masked=False)(x, x, x)
        x = nn.LayerNorm()(x + attn_out)

        mlp_out = nn.Dense(self.n_embd, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(self.n_embd, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(mlp_out)
        return nn.LayerNorm()(x + mlp_out)


class DecodeBlock(nn.Module):
    n_embd: int
    n_head: int

    @nn.compact
    def __call__(self, x: jax.Array, rep_enc: jax.Array) -> jax.Array:
        attn1_out = MultiHeadAttention(self.n_embd, self.n_head, masked=True)(x, x, x)
        x = nn.LayerNorm()(x + attn1_out)

        attn2_out = MultiHeadAttention(self.n_embd, self.n_head, masked=True)(
            query=rep_enc, key=x, value=x
        )
        x = nn.LayerNorm()(rep_enc + attn2_out)

        mlp_out = nn.Dense(self.n_embd, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        mlp_out = nn.gelu(mlp_out)
        mlp_out = nn.Dense(self.n_embd, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(mlp_out)
        return nn.LayerNorm()(x + mlp_out)


class MATEncoder(nn.Module):
    n_embd: int
    n_head: int
    n_blocks: int

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        # obs: (batch, n_agents, obs_dim)
        x = nn.LayerNorm()(obs)
        x = nn.Dense(self.n_embd, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        for _ in range(self.n_blocks):
            x = EncodeBlock(self.n_embd, self.n_head)(x)
        rep = x

        v = nn.Dense(self.n_embd, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(rep)
        v = nn.gelu(v)
        v = nn.LayerNorm()(v)
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(v)
        v = jnp.squeeze(v, axis=-1)  # (batch, n_agents)

        return v, rep


class MATDecoder(nn.Module):
    action_dim: int
    n_embd: int
    n_head: int
    n_blocks: int

    @nn.compact
    def __call__(self, shifted_actions: jax.Array, rep_enc: jax.Array) -> jax.Array:
        # shifted_actions: (batch, n_agents, action_dim + 1)
        x = nn.Dense(self.n_embd, use_bias=False, kernel_init=orthogonal(np.sqrt(2)))(shifted_actions)
        x = nn.gelu(x)
        x = nn.LayerNorm()(x)

        for _ in range(self.n_blocks):
            x = DecodeBlock(self.n_embd, self.n_head)(x, rep_enc)

        logits = nn.Dense(self.n_embd, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        logits = nn.gelu(logits)
        logits = nn.LayerNorm()(logits)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(logits)
        return logits


class MATModel(nn.Module):
    action_dim: int
    n_agents: int
    n_embd: int = 64
    n_head: int = 1
    n_blocks: int = 1

    def setup(self):
        self.encoder = MATEncoder(self.n_embd, self.n_head, self.n_blocks)
        self.decoder = MATDecoder(self.action_dim, self.n_embd, self.n_head, self.n_blocks)

    def evaluate_actions(
        self, obs: jax.Array, actions: jax.Array, avail_actions: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        # obs: (batch, n_agents, obs_dim)
        # actions: (batch, n_agents)
        # avail_actions: (batch, n_agents, action_dim)
        batch_size = obs.shape[0]
        v_loc, rep = self.encoder(obs)

        one_hot_act = jax.nn.one_hot(actions, self.action_dim)
        shifted = jnp.zeros((batch_size, self.n_agents, self.action_dim + 1), dtype=jnp.float32)
        shifted = shifted.at[:, 0, 0].set(1.0)
        shifted = shifted.at[:, 1:, 1:].set(one_hot_act[:, :-1, :])

        logits = self.decoder(shifted, rep)
        logits = logits - (1.0 - avail_actions) * 1e10

        pi = distrax.Categorical(logits=logits)
        log_prob = pi.log_prob(actions)
        entropy = pi.entropy()
        return log_prob, v_loc, entropy

    def get_values(self, obs: jax.Array) -> jax.Array:
        v_loc, _ = self.encoder(obs)
        return v_loc

    def get_actions(
        self, rng: jax.Array, obs: jax.Array, avail_actions: jax.Array
    ) -> tuple[jax.Array, jax.Array, jax.Array]:
        batch_size = obs.shape[0]
        v_loc, rep = self.encoder(obs)

        # Autoregressive generation over n_agents
        shifted = jnp.zeros((batch_size, self.n_agents, self.action_dim + 1), dtype=jnp.float32)
        shifted = shifted.at[:, 0, 0].set(1.0)

        actions_list = []
        log_probs_list = []

        for i in range(self.n_agents):
            rng, rng_act = jax.random.split(rng)
            logits = self.decoder(shifted, rep)
            logit_i = logits[:, i, :]
            mask_i = avail_actions[:, i, :]
            logit_i = logit_i - (1.0 - mask_i) * 1e10

            pi_i = distrax.Categorical(logits=logit_i)
            action_i = pi_i.sample(seed=rng_act)
            log_prob_i = pi_i.log_prob(action_i)

            actions_list.append(action_i)
            log_probs_list.append(log_prob_i)

            if i + 1 < self.n_agents:
                one_hot_i = jax.nn.one_hot(action_i, self.action_dim)
                shifted = shifted.at[:, i + 1, 1:].set(one_hot_i)

        output_actions = jnp.stack(actions_list, axis=1)
        output_log_probs = jnp.stack(log_probs_list, axis=1)
        return output_actions, output_log_probs, v_loc


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
        model = MATModel(
            action_dim=action_dim,
            n_agents=num_agents,
            n_embd=config["N_EMBD"],
            n_head=config["N_HEAD"],
            n_blocks=config["N_BLOCKS"],
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

        def _log_update(metrics, update_index):
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

                actions, log_probs, values = model.apply(
                    train_state.params,
                    rng=rng_act,
                    obs=last_obs,
                    avail_actions=avail_actions,
                    method=model.get_actions,
                )

                env_act = {
                    agent: actions[:, i] for i, agent in enumerate(env.agents)
                }

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

                # Normalize advantage across batch
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

                        # Policy loss
                        ratio = jnp.exp(log_prob - mb.log_prob)
                        surr1 = ratio * mb_adv
                        surr2 = jnp.clip(ratio, 1.0 - config["CLIP_EPS"], 1.0 + config["CLIP_EPS"]) * mb_adv
                        actor_loss = -jnp.mean(jnp.minimum(surr1, surr2))

                        # Value loss (clipped)
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

            # Metrics callback for WandB
            metrics = {
                "loss": jnp.mean(loss_metrics[0]),
                "returned_won_episode": jnp.mean(
                    traj_batch.info["returned_won_episode"]
                ),
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


@hydra.main(version_base=None, config_path="config", config_name="mat_smax")
def main(config):
    config = OmegaConf.to_container(config)
    wandb.init(
        entity=config.get("ENTITY", None),
        project=config["WANDB_PROJECT"],
        tags=["MAT", "SMAX", config["MAP_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config.get("SEED", 42))
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    wandb.finish()
    print("MAT training on SMAX 3s5z completed successfully!")


if __name__ == "__main__":
    main()
