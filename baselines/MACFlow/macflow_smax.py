"""MAC-Flow (Multi-Agent Continuous Normalizing Flow / Flow Matching) in 100% JAX/Flax for SMAX.

Based on:
- 'Multi-Agent Continuous Normalizing Flow / Flow Matching' (NeurIPS/ICLR)
- Reference: temp/mac-flow/agents/discrete_macflow.py
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
from baselines.metrics import (
    finished_episode_stats,
    greedy_eval_rollout_multiseed,
    log_metrics,
)
from jaxmarl import make
from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import SMAXLogWrapper


# =============================================================================
# MAC-Flow Architecture
# =============================================================================

def sinusoidal_time_embed(t: jax.Array, num_freqs: int = 8) -> jax.Array:
    """Sinusoidal time embedding for flow time t in [0, 1]."""
    freqs = jnp.array([2.0 ** i for i in range(num_freqs)], dtype=t.dtype) * jnp.pi
    ang = t * freqs
    return jnp.concatenate([jnp.sin(ang), jnp.cos(ang)], axis=-1)


class VectorFieldNet(nn.Module):
    action_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs: jax.Array, x: jax.Array, t_embed: jax.Array) -> jax.Array:
        # Concatenate observation, state x_t, and time embedding
        inputs = jnp.concatenate([obs, x, t_embed], axis=-1)
        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(inputs)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
        h = nn.gelu(h)
        v = nn.Dense(self.action_dim, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h)
        return v


class OneStepFlowNet(nn.Module):
    action_dim: int
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, obs: jax.Array, noise: jax.Array) -> jax.Array:
        # One-step flow predictor mapping observation and initial noise to action logits
        inputs = jnp.concatenate([obs, noise], axis=-1)
        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(inputs)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
        h = nn.gelu(h)
        logits = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(h)
        return logits


class CentralizedCritic(nn.Module):
    hidden_dim: int = 128

    @nn.compact
    def __call__(self, flat_obs: jax.Array) -> jax.Array:
        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(flat_obs)
        h = nn.gelu(h)
        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
        h = nn.gelu(h)
        v = nn.Dense(1, kernel_init=orthogonal(1.0), bias_init=constant(0.0))(h)
        return v


class MACFlowModel(nn.Module):
    action_dim: int
    n_agents: int
    hidden_dim: int = 128

    def setup(self):
        self.vf_net = VectorFieldNet(action_dim=self.action_dim, hidden_dim=self.hidden_dim)
        self.onestep_net = OneStepFlowNet(action_dim=self.action_dim, hidden_dim=self.hidden_dim)
        self.critic = CentralizedCritic(hidden_dim=self.hidden_dim)

    def vector_field(self, obs: jax.Array, x: jax.Array, t_embed: jax.Array) -> jax.Array:
        return self.vf_net(obs, x, t_embed)

    def onestep_flow(self, obs: jax.Array, noise: jax.Array) -> jax.Array:
        return self.onestep_net(obs, noise)

    def value(self, flat_obs: jax.Array) -> jax.Array:
        return self.critic(flat_obs)

    def init_all(self, obs: jax.Array, avail_actions: jax.Array, noise: jax.Array):
        pi, value = self(obs, avail_actions, noise)
        t_dummy = sinusoidal_time_embed(jnp.zeros((*obs.shape[:-1], 1)))
        _ = self.vf_net(obs, noise, t_dummy)
        return pi, value

    def __call__(
        self,
        obs: jax.Array,
        avail_actions: jax.Array,
        noise: jax.Array,
    ) -> tuple[distrax.Categorical, jax.Array]:
        batch_size, n_agents, _ = obs.shape
        logits = self.onestep_net(obs, noise)
        large_neg = -1e10
        masked_logits = jnp.where(avail_actions > 0.5, logits, large_neg)
        pi = distrax.Categorical(logits=masked_logits)

        flat_obs = obs.reshape(batch_size, -1)
        v = self.critic(flat_obs)
        value = jnp.tile(v, (1, n_agents))
        return pi, value

    def get_greedy_actions(
        self, obs: jax.Array, avail_actions: jax.Array
    ) -> jax.Array:
        dummy_noise = jnp.zeros((*obs.shape[:-1], self.action_dim))
        pi, _ = self(obs, avail_actions, dummy_noise)
        return jnp.argmax(pi.logits, axis=-1)


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
        model = MACFlowModel(
            action_dim=action_dim,
            n_agents=num_agents,
            hidden_dim=config["HIDDEN_DIM"],
        )

        rng, rng_init = jax.random.split(rng)
        dummy_obs = jnp.zeros((config["NUM_ENVS"], num_agents, obs_dim))
        dummy_avail = jnp.ones((config["NUM_ENVS"], num_agents, action_dim))
        dummy_noise = jnp.zeros((config["NUM_ENVS"], num_agents, action_dim))

        params = model.init(
            rng_init,
            obs=dummy_obs,
            avail_actions=dummy_avail,
            noise=dummy_noise,
            method=model.init_all,
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
            log_metrics(
                {key: float(value) for key, value in metrics.items()},
                step=int(update_index)
                * int(config["NUM_STEPS"])
                * int(config["NUM_ENVS"]),
            )

        # 3. Main Update Loop
        def _update_step(runner_state, update_index):
            train_state, env_state, last_obs, last_done, rng, test_state = runner_state

            # --- ROLLOUT ---
            def _env_step(carry, _):
                env_state, last_obs, last_done, rng = carry
                rng, rng_noise, rng_act, rng_step = jax.random.split(rng, 4)

                avail_actions_dict = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = dict_to_agent_array(avail_actions_dict, env.agents)

                noise = jax.random.normal(rng_noise, (*last_obs.shape[:2], action_dim))
                pi, value = model.apply(
                    train_state.params,
                    obs=last_obs,
                    avail_actions=avail_actions,
                    noise=noise,
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
            dummy_noise = jnp.zeros((*last_obs.shape[:2], action_dim))
            _, last_val = model.apply(
                train_state.params,
                obs=last_obs,
                avail_actions=avail_actions,
                noise=dummy_noise,
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

            # --- PPO & FLOW MATCHING UPDATE ---
            def _update_epoch(update_state, _):
                def _update_minibatch(train_state, batch_info):
                    traj_batch, advantages, targets, batch_rng = batch_info

                    def _loss_fn(params):
                        # 1. PPO Policy & Value Loss
                        noise_eval = jnp.zeros((*traj_batch.obs.shape[:-1], action_dim))
                        pi, value = model.apply(
                            params,
                            obs=traj_batch.obs,
                            avail_actions=traj_batch.avail_actions,
                            noise=noise_eval,
                        )
                        log_prob = pi.log_prob(traj_batch.action)

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

                        v_clipped = traj_batch.value + jnp.clip(
                            value - traj_batch.value,
                            -config["CLIP_EPS"],
                            config["CLIP_EPS"],
                        )
                        v_loss1 = jnp.square(value - targets)
                        v_loss2 = jnp.square(v_clipped - targets)
                        critic_loss = 0.5 * jnp.maximum(v_loss1, v_loss2).mean()

                        entropy = pi.entropy().mean()

                        # 2. Flow Matching Loss
                        rng_x0, rng_t, rng_distill = jax.random.split(batch_rng, 3)
                        x_1 = jax.nn.one_hot(traj_batch.action, action_dim)
                        x_0 = jax.random.normal(rng_x0, x_1.shape)
                        t_scalar = jax.random.uniform(rng_t, (*x_1.shape[:-1], 1))
                        x_t = (1.0 - t_scalar) * x_0 + t_scalar * x_1
                        target_vel = x_1 - x_0

                        t_embed = sinusoidal_time_embed(t_scalar, num_freqs=config["T_EMBED_FREQUENCIES"])
                        pred_vel = model.apply(
                            params,
                            obs=traj_batch.obs,
                            x=x_t,
                            t_embed=t_embed,
                            method=model.vector_field,
                        )
                        bc_flow_loss = jnp.mean(jnp.square(pred_vel - target_vel))

                        # 3. Flow Distillation Loss (one-step model distills target actions)
                        noises_distill = jax.random.normal(rng_distill, x_1.shape)
                        onestep_logits = model.apply(
                            params,
                            obs=traj_batch.obs,
                            noise=noises_distill,
                            method=model.onestep_flow,
                        )
                        distill_loss = jnp.mean(
                            optax.softmax_cross_entropy_with_integer_labels(
                                logits=onestep_logits,
                                labels=traj_batch.action,
                            )
                        )

                        total_loss = (
                            actor_loss
                            + config["VF_COEF"] * critic_loss
                            - config["ENT_COEF"] * entropy
                            + bc_flow_loss
                            + config["ALPHA"] * distill_loss
                        )
                        return total_loss, (actor_loss, critic_loss, entropy, bc_flow_loss, distill_loss)

                    grad_fn = jax.value_and_grad(_loss_fn, has_aux=True)
                    (loss, (a_loss, c_loss, ent, flow_l, dist_l)), grads = grad_fn(train_state.params)
                    train_state = train_state.apply_gradients(grads=grads)
                    return train_state, (loss, a_loss, c_loss, ent, flow_l, dist_l)

                train_state, traj_batch, advantages, targets, rng = update_state
                rng, rng_perm, rng_minibatch = jax.random.split(rng, 3)

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

                # Split random keys for minibatches
                minibatch_rngs = jax.random.split(rng_minibatch, config["NUM_MINIBATCHES"])
                full_minibatch_info = (*minibatches, minibatch_rngs)

                train_state, (loss, a_loss, c_loss, ent, flow_l, dist_l) = jax.lax.scan(
                    _update_minibatch, train_state, full_minibatch_info
                )
                return (train_state, traj_batch, advantages, targets, rng), (
                    loss,
                    a_loss,
                    c_loss,
                    ent,
                    flow_l,
                    dist_l,
                )

            update_state = (train_state, traj_batch, advantages, targets, rng)
            update_state, (loss, a_loss, c_loss, ent, flow_l, dist_l) = jax.lax.scan(
                _update_epoch, update_state, None, length=config["UPDATE_EPOCHS"]
            )
            train_state = update_state[0]
            rng = update_state[-1]

            # Periodic greedy evaluation (epsilon = 0) during training
            eval_interval = max(
                1, int(config.get("TEST_INTERVAL", 0.05) * config["NUM_UPDATES"])
            )

            def _run_eval(rng):
                def greedy_fn(obs, avail):
                    return model.apply(
                        train_state.params,
                        obs=obs,
                        avail_actions=avail,
                        method=model.get_greedy_actions,
                    )

                return greedy_eval_rollout_multiseed(
                    env=env,
                    agents=env.agents,
                    num_agents=num_agents,
                    rng=rng,
                    n_test_seeds=int(config.get("TEST_NUM_SEEDS", 5)),
                    n_env=int(config.get("TEST_NUM_ENVS", 128)),
                    n_steps=int(config.get("TEST_NUM_STEPS", 128)),
                    greedy_fn=greedy_fn,
                    dict_to_agent_array=dict_to_agent_array,
                )

            rng, rng_eval = jax.random.split(rng)
            new_test_state = jax.lax.cond(
                (update_index + 1) % eval_interval == 0,
                lambda: _run_eval(rng_eval),
                lambda: test_state,
            )

            # --- METRICS & LOGGING ---
            metrics = {
                "loss": loss.mean(),
                "actor_loss": a_loss.mean(),
                "critic_loss": c_loss.mean(),
                "entropy": ent.mean(),
                "bc_flow_loss": flow_l.mean(),
                "distill_loss": dist_l.mean(),
                "returned_won_episode": jnp.mean(traj_batch.info["returned_won_episode"]),
                "returned_episode_returns": jnp.mean(
                    traj_batch.info["returned_episode_returns"]
                ),
                "returned_episode_lengths": jnp.mean(
                    traj_batch.info["returned_episode_lengths"]
                ),
                "returned_episode": jnp.mean(traj_batch.info["returned_episode"]),
            }
            for k in test_template:
                arr = new_test_state[k]
                metrics[f"test_{k}/mean"] = jnp.mean(arr)
                metrics[f"test_{k}/std"] = jnp.std(arr)
            jax.debug.callback(_log_update, metrics, update_index + 1)

            runner_state = (train_state, env_state, last_obs, last_done, rng, new_test_state)
            return runner_state, metrics

        # 4. Run All Updates via lax.scan
        n_test_seeds = int(config.get("TEST_NUM_SEEDS", 5))
        test_template = {
            "returned_won_episode": jnp.zeros((n_test_seeds,)),
            "returned_episode_returns": jnp.zeros((n_test_seeds,)),
            "returned_episode_lengths": jnp.zeros((n_test_seeds,)),
            "returned_episode": jnp.zeros((n_test_seeds,)),
        }
        runner_state = (
            train_state,
            env_state,
            last_obs,
            last_done,
            rng,
            test_template,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )
        return {"runner_state": runner_state, "metrics": metrics}

    return train


@hydra.main(version_base=None, config_path="config", config_name="macflow_smax")
def main(config):
    config = OmegaConf.to_container(config)

    wandb.init(
        entity=config.get("ENTITY", None),
        project=config["WANDB_PROJECT"],
        tags=["MACFLOW", "SMAX", config["MAP_NAME"]],
        config=config,
        mode=config["WANDB_MODE"],
    )

    rng = jax.random.PRNGKey(config.get("SEED", 42))
    train_jit = jax.jit(make_train(config))
    out = train_jit(rng)
    final = out["metrics"]
    print(
        f"MAC-Flow on {config['MAP_NAME']} | "
        f"test win rate: {float(final['test_returned_won_episode/mean'][-1]):.3f} ± "
        f"{float(final['test_returned_won_episode/std'][-1]):.3f} | "
        f"test return: {float(final['test_returned_episode_returns/mean'][-1]):.3f} ± "
        f"{float(final['test_returned_episode_returns/std'][-1]):.3f}"
    )
    wandb.finish()


if __name__ == "__main__":
    main()
