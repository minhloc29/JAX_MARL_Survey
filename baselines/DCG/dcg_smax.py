"""Deep Coordination Graphs (DCG) in JAX/Flax for SMAX (revised).

Based on 'Deep Coordination Graphs' (Boehmer et al., ICML 2020).

Changes versus the first port:
  * Pairwise payoffs are symmetrized the way the paper does it: the SAME
    payoff network is applied in both input orders and the results are
    averaged (f_ij(a_i, a_j) = 0.5 * (g(h_i, h_j)[a_i, a_j] + g(h_j, h_i)[a_j, a_i])),
    instead of forcing f_ij == f_ij^T.
  * One-hot agent ID appended to observations (config ADD_AGENT_ID, default True).
  * Win rate / return / length are averaged over FINISHED episodes only.
  * Greedy (epsilon = 0) frozen-policy evaluation at the end of training.
  * Double-Q target is computed outside the loss (no autodiff through
    message passing).
  * Training returns only metrics, eval and params (not the replay buffer), so
    jax.vmap over seeds does not keep one buffer per seed on the device.
  * Shared metric vocabulary via baselines.metrics.log_metrics, optional in-loop
    logging (config LOG_IN_LOOP) so seeds can be vmapped.

Known deviations from the original paper (list them in the deviations table):
  * No recurrent agent network (MLP on the current observation only).
  * Full |A| x |A| payoff matrices, not the low-rank factorization.
  * Complete graph over agents (no sparsity / no learned graph).
  * Time-limit truncation is treated as terminal.
"""

from __future__ import annotations

from typing import Any

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
from baselines.metrics import (
    finished_episode_stats,
    greedy_eval_rollout,
    greedy_eval_rollout_multiseed,
    log_metrics,
)
from jaxmarl.environments.smax import HeuristicEnemySMAX, map_name_to_scenario
from jaxmarl.wrappers.baselines import SMAXLogWrapper


# =============================================================================
# Helpers
# =============================================================================

def make_edges(n_agents: int) -> tuple[jax.Array, jax.Array]:
    """Undirected complete graph, each pair (i < j) once."""
    pairs = [(i, j) for i in range(n_agents) for j in range(i + 1, n_agents)]
    edges_from = jnp.array([p[0] for p in pairs], dtype=jnp.int32)
    edges_to = jnp.array([p[1] for p in pairs], dtype=jnp.int32)
    return edges_from, edges_to


def dict_to_agent_array(d, agent_list):
    return jnp.stack([d[agent] for agent in agent_list], axis=1)


# =============================================================================
# DCG Networks (Flax Linen)
# =============================================================================

class PairPayoff(nn.Module):
    """g(h_a, h_b) -> payoff matrix indexed [action_of_a, action_of_b]."""

    hidden_dim: int
    action_dim: int

    @nn.compact
    def __call__(self, h_a: jax.Array, h_b: jax.Array) -> jax.Array:
        x = jnp.concatenate([h_a, h_b], axis=-1)
        x = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(x)
        x = nn.relu(x)
        x = nn.Dense(self.action_dim * self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(x)
        return x.reshape(x.shape[:-1] + (self.action_dim, self.action_dim))


class DCGNet(nn.Module):
    action_dim: int
    n_agents: int
    hidden_dim: int = 128
    add_agent_id: bool = True

    @nn.compact
    def __call__(self, obs: jax.Array) -> tuple[jax.Array, jax.Array]:
        # obs: (batch, n_agents, obs_dim)
        if self.add_agent_id:
            ids = jnp.broadcast_to(
                jnp.eye(self.n_agents), obs.shape[:-2] + (self.n_agents, self.n_agents)
            )
            obs = jnp.concatenate([obs, ids], axis=-1)

        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(obs)
        h = nn.relu(h)
        h = nn.Dense(self.hidden_dim, kernel_init=orthogonal(np.sqrt(2)), bias_init=constant(0.0))(h)
        h = nn.relu(h)  # (batch, n_agents, hidden_dim)

        # Unary utilities f_i
        f_i = nn.Dense(self.action_dim, kernel_init=orthogonal(0.01), bias_init=constant(0.0))(h)

        # Pairwise payoffs: one shared network, applied in both input orders
        edges_from, edges_to = make_edges(self.n_agents)
        h_from = h[:, edges_from, :]
        h_to = h[:, edges_to, :]
        payoff = PairPayoff(self.hidden_dim, self.action_dim, name="payoff")
        p_fwd = payoff(h_from, h_to)  # [a_from, a_to]
        p_bwd = payoff(h_to, h_from)  # [a_to, a_from]
        f_ij = 0.5 * (p_fwd + jnp.swapaxes(p_bwd, -2, -1))  # [a_from, a_to]

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
    """Max-plus message passing; maximizes Q_tot = mean_i f_i + mean_edges f_ij."""
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
# Training
# =============================================================================

class CustomTrainState(TrainState):
    target_params: Any
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
    edges_from, edges_to = make_edges(num_agents)

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
        model = DCGNet(
            action_dim=action_dim,
            n_agents=num_agents,
            hidden_dim=config["HIDDEN_DIM"],
            add_agent_id=config.get("ADD_AGENT_ID", True),
        )

        rng, rng_init = jax.random.split(rng)
        dummy_obs = jnp.zeros((config["NUM_ENVS"], num_agents, obs_dim))
        params = model.init(rng_init, dummy_obs)

        tx = optax.chain(
            optax.clip_by_global_norm(config["MAX_GRAD_NORM"]),
            optax.adam(learning_rate=config["LR"], eps=1e-5),
        )
        train_state = CustomTrainState.create(
            apply_fn=model.apply, params=params, target_params=params, tx=tx
        )

        # Replay buffer (Flashbax flat buffer; add whole rollouts at once)
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

        rng, rng_reset = jax.random.split(rng)
        reset_rngs = jax.random.split(rng_reset, config["NUM_ENVS"])
        obs_dict, env_state = jax.vmap(env.reset)(reset_rngs)
        last_obs = dict_to_agent_array(obs_dict, env.agents)
        last_done = jnp.zeros(config["NUM_ENVS"], dtype=jnp.bool_)

        def _log_update(metrics, update_index):
            log_metrics(
                {key: float(value) for key, value in metrics.items()},
                step=int(update_index)
                * int(config["NUM_STEPS"])
                * int(config["NUM_ENVS"]),
            )

        def _greedy(params, obs, avail):
            f_i, f_ij = model.apply(params, obs)
            actions, _ = message_passing(
                f_i, f_ij, edges_from, edges_to, avail,
                num_agents, action_dim, iterations=config["MSG_ITERATIONS"],
            )
            return actions

        def _update_step(runner_state, update_idx):
            train_state, buffer_state, env_state, last_obs, last_done, rng, test_state = runner_state
            eps = eps_scheduler(train_state.n_updates)

            # --- COLLECT ROLLOUT (epsilon-greedy) ---
            def _env_step(carry, _):
                env_state, last_obs, last_done, rng = carry
                rng, rng_eps, rng_rand, rng_step = jax.random.split(rng, 4)

                avail_dict = jax.vmap(env.get_avail_actions)(env_state.env_state)
                avail_actions = dict_to_agent_array(avail_dict, env.agents).astype(jnp.float32)

                greedy_act = _greedy(train_state.params, last_obs, avail_actions)

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
                return (env_state, next_obs, global_done, rng), (trans, infos)

            carry = (env_state, last_obs, last_done, rng)
            carry, (traj_trans, infos) = jax.lax.scan(_env_step, carry, None, config["NUM_STEPS"])
            env_state, last_obs, last_done, rng = carry

            flat_trans = jax.tree.map(
                lambda x: x.reshape((config["NUM_STEPS"] * config["NUM_ENVS"],) + x.shape[2:]),
                traj_trans,
            )
            buffer_state = buffer.add(buffer_state, flat_trans)

            # --- DOUBLE-Q UPDATES ---
            def _epoch_step(carry, _):
                train_state, buffer_state, rng = carry
                rng, rng_sample = jax.random.split(rng)
                batch = buffer.sample(buffer_state, rng_sample).experience.first

                # Target: online net selects next joint action, target net evaluates it.
                # Computed outside the loss, so no gradient flows through message passing.
                greedy_next = _greedy(train_state.params, batch.next_obs, batch.next_avail)
                t_f_i, t_f_ij = model.apply(train_state.target_params, batch.next_obs)
                q_next = compute_q_tot(t_f_i, t_f_ij, greedy_next, edges_from, edges_to)
                team_reward = jnp.mean(batch.rewards, axis=-1)
                not_done = 1.0 - batch.dones.astype(jnp.float32)
                target_y = jax.lax.stop_gradient(
                    team_reward + config["GAMMA"] * not_done * q_next
                )

                def _loss_fn(params):
                    f_i, f_ij = model.apply(params, batch.obs)
                    q_curr = compute_q_tot(f_i, f_ij, batch.actions, edges_from, edges_to)
                    return jnp.mean(jnp.square(q_curr - target_y))

                loss, grads = jax.value_and_grad(_loss_fn)(train_state.params)
                train_state = train_state.apply_gradients(grads=grads)
                return (train_state, buffer_state, rng), loss

            (train_state, buffer_state, rng), losses = jax.lax.scan(
                _epoch_step, (train_state, buffer_state, rng), None, config["NUM_EPOCHS"]
            )

            new_target = jax.tree.map(
                lambda p, tp: jnp.where((update_idx % config["TARGET_UPDATE_INTERVAL"]) == 0, p, tp),
                train_state.params,
                train_state.target_params,
            )
            train_state = train_state.replace(
                target_params=new_target, n_updates=train_state.n_updates + 1
            )

            metrics = finished_episode_stats(infos, num_agents)
            metrics["loss"] = jnp.mean(losses)
            metrics["epsilon"] = eps

            # Periodic greedy evaluation (epsilon = 0) during training, matching RODE.
            eval_interval = max(
                1, int(config.get("TEST_INTERVAL", 0.05) * config["NUM_UPDATES"])
            )

            def _run_eval(rng):
                def greedy_fn(obs, avail):
                    return _greedy(train_state.params, obs, avail)

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
                (update_idx + 1) % eval_interval == 0,
                lambda: _run_eval(rng_eval),
                lambda: test_state,
            )
            # Log per-test-seed mean and std so WandB shows both the value and
            # the run-to-run variance, plus individual seed traces.
            for k in test_template:
                arr = new_test_state[k]
                metrics[f"test_{k}/mean"] = jnp.mean(arr)
                metrics[f"test_{k}/std"] = jnp.std(arr)

            if config.get("LOG_IN_LOOP", True):
                jax.debug.callback(_log_update, metrics, update_idx + 1)

            runner_state = (
                train_state,
                buffer_state,
                env_state,
                last_obs,
                last_done,
                rng,
                new_test_state,
            )
            return runner_state, metrics

        n_test_seeds = int(config.get("TEST_NUM_SEEDS", 5))
        test_template = {
            "returned_won_episode": jnp.zeros((n_test_seeds,)),
            "returned_episode_returns": jnp.zeros((n_test_seeds,)),
            "returned_episode_lengths": jnp.zeros((n_test_seeds,)),
            "returned_episode": jnp.zeros((n_test_seeds,)),
        }
        runner_state = (
            train_state,
            buffer_state,
            env_state,
            last_obs,
            last_done,
            rng,
            test_template,
        )
        runner_state, metrics = jax.lax.scan(
            _update_step, runner_state, jnp.arange(config["NUM_UPDATES"])
        )

        # Do not return the replay buffer (large, and one copy per seed under vmap).
        return {"metrics": metrics, "params": runner_state[0].params}

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
    out = jax.jit(make_train(config))(rng)
    final = out["metrics"]
    print(
        f"DCG on {config['MAP_NAME']} | "
        f"test win rate: {float(final['test_returned_won_episode/mean'][-1]):.3f} ± "
        f"{float(final['test_returned_won_episode/std'][-1]):.3f} | "
        f"test return: {float(final['test_returned_episode_returns/mean'][-1]):.3f} ± "
        f"{float(final['test_returned_episode_returns/std'][-1]):.3f}"
    )
    wandb.finish()


if __name__ == "__main__":
    main()