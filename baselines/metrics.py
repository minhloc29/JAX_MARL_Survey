"""W&B logging helpers that preserve environment-provided metric names."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import wandb


def log_metrics(metrics: Mapping[str, Any], *, step: int | None = None) -> None:
    """Log metrics without renaming or synthesizing metric fields."""
    wandb.log(dict(metrics), step=step)


def finished_episode_stats(infos: Mapping[str, jax.Array], num_agents: int) -> dict:
    """Aggregate SMAXLogWrapper ``returned_*`` fields over *finished* episodes only.

    The :class:`SMAXLogWrapper` repeats the value of the most recent finished
    episode on every step, only writing a fresh value on steps where an episode
    just ended (``returned_episode`` becomes 1). A plain ``mean`` over the rollout
    is therefore step-weighted and dominated by the longest episodes. Weighting by
    the ``returned_episode`` mask gives each *finished episode* equal weight, which
    is the per-episode win rate / return an evaluation reports.

    ``infos`` leaves are expected to have shape ``(..., num_agents)``. Output keys
    mirror the wrapper vocabulary: ``returned_won_episode``, ``returned_episode_returns``,
    ``returned_episode_lengths``, ``returned_episode`` (fraction of finished episodes).
    """
    done_ep = infos["returned_episode"].astype(jnp.float32)
    n = jnp.maximum(done_ep.sum(), 1.0)
    return {
        "returned_won_episode": (infos["returned_won_episode"] * done_ep).sum() / n,
        "returned_episode_returns": (infos["returned_episode_returns"] * done_ep).sum() / n,
        "returned_episode_lengths": (infos["returned_episode_lengths"] * done_ep).sum() / n,
        "returned_episode": done_ep.sum() / num_agents,
    }


def greedy_eval_rollout(
    *,
    env,
    agents: list[str],
    num_agents: int,
    rng: jax.Array,
    n_env: int,
    n_steps: int,
    greedy_fn,
    dict_to_agent_array,
) -> dict:
    """Run a frozen (epsilon=0) greedy rollout and return finished-episode stats.

    This is the shared implementation of RODE's ``get_greedy_metrics`` for the
    "dict-env / agent-array" baseline family (MAT, DCG, DICG, IPPO, MAPPO). A
    greedy evaluation isolates the learned policy from training-time exploration,
    and finished-episode stats give an episode-weighted win rate / return.

    Args:
        env: a wrapped env exposing ``reset``/``step`` (``SMAXLogWrapper``) and
            per-agent ``get_avail_actions`` on ``env_state.env_state``.
        agents: ordered agent names.
        num_agents: number of agents.
        rng: PRNG key for the evaluation.
        n_env: number of parallel evaluation environments.
        n_steps: rollout length.
        greedy_fn: callable ``(obs, avail_actions) -> actions`` with shape
            ``(n_env, num_agents)``, where actions are greedily selected. obs and
            avail have shape ``(n_env, num_agents, dim)``.
        dict_to_agent_array: ``(dict, agent_list) -> (n_env, num_agents, dim)``.
    """
    rng, rng_reset = jax.random.split(rng)
    obs_dict, env_state = jax.vmap(env.reset)(jax.random.split(rng_reset, n_env))
    obs = dict_to_agent_array(obs_dict, agents)

    def _eval_step(carry, _):
        env_state, obs, rng = carry
        rng, rng_step, rng_avail = jax.random.split(rng, 3)
        avail_dict = jax.vmap(env.get_avail_actions)(env_state.env_state)
        avail = dict_to_agent_array(avail_dict, agents).astype(jnp.float32)
        actions = greedy_fn(obs, avail)
        env_act = {agent: actions[:, i] for i, agent in enumerate(agents)}
        step_rngs = jax.random.split(rng_step, n_env)
        next_obs_dict, env_state, _, _, infos = jax.vmap(env.step)(
            step_rngs, env_state, env_act
        )
        next_obs = dict_to_agent_array(next_obs_dict, agents)
        return (env_state, next_obs, rng), infos

    _, infos = jax.lax.scan(_eval_step, (env_state, obs, rng), None, n_steps)
    return finished_episode_stats(infos, num_agents)


def test_means_stds(per_seed: Mapping[str, jax.Array]) -> dict[str, jax.Array]:
    """Map per-seed eval stats to a ``test_<stat>/mean`` + ``test_<stat>/std`` dict.

    ``per_seed`` maps each stat (e.g. ``returned_won_episode``) to an array of
    shape ``(n_test_seeds,)``. Returns flat keys ready for ``wandb.log``:
    ``test_returned_won_episode/mean`` and ``test_returned_won_episode/std``.
    This is the shared convention across all JAX baselines.
    """
    out: dict[str, jax.Array] = {}
    for k, v in per_seed.items():
        out[f"test_{k}/mean"] = jnp.mean(v)
        out[f"test_{k}/std"] = jnp.std(v)
    return out


def eval_multiseed_scan(
    single_eval, n_test_seeds: int, rng: jax.Array
) -> dict[str, jax.Array]:
    """Run ``single_eval`` over ``n_test_seeds`` independent PRNG keys.

    ``single_eval`` is any pure callable mapping a scalar PRNG key to a dict of
    scalars (e.g. a closure around an algorithm's ``get_greedy_metrics(rng, ...)``).
    This runs it sequentially over seeds via ``lax.scan``, which composes correctly
    even when ``single_eval`` itself ``vmap``-batches many environments internally
    (a naive ``jax.vmap`` over the seed axis would collide with that inner axis).

    Returns:
        dict mapping each output key to an array of shape ``(n_test_seeds,)``.
        Reduce with :func:`eval_means_stds` for mean/std logging on WandB.
    """
    rngs = jax.random.split(rng, n_test_seeds)

    def _one(_, seed_rng):
        return None, single_eval(seed_rng)

    _, per_seed = jax.lax.scan(_one, None, rngs)
    return per_seed


def greedy_eval_rollout_multiseed(
    *,
    env,
    agents: list[str],
    num_agents: int,
    rng: jax.Array,
    n_test_seeds: int,
    n_env: int,
    n_steps: int,
    greedy_fn,
    dict_to_agent_array,
) -> dict:
    """Run ``n_test_seeds`` independent (frozen, epsilon=0) greedy evals.

    A single :func:`greedy_eval_rollout` folds ``n_env`` parallel environments
    into one episode-weighted metric; that averages out the *environment* noise
    but gives no sense of run-to-run (seed) variance. This helper runs
    ``n_test_seeds`` separate greedy rollouts, each with its own split PRNG key,
    and returns one per-seed entry per finished-episode stat.

    Returns:
        dict mapping each stat key (``returned_won_episode``,
        ``returned_episode_returns``, ``returned_episode_lengths``,
        ``returned_episode``) to an array of shape ``(n_test_seeds,)``. Reduce with
        ``.mean()`` / ``.std()`` for aggregate logging on WandB. ``rng`` is split
        deterministically, so a fixed training seed reproduces the same test seeds.
    """
    rngs = jax.random.split(rng, n_test_seeds)

    def _one(_, seed_rng):
        stats = greedy_eval_rollout(
            env=env,
            agents=agents,
            num_agents=num_agents,
            rng=seed_rng,
            n_env=n_env,
            n_steps=n_steps,
            greedy_fn=greedy_fn,
            dict_to_agent_array=dict_to_agent_array,
        )
        return None, stats

    _, per_seed = jax.lax.scan(_one, None, rngs)
    return per_seed


def greedy_eval_rollout_stateful(
    *,
    env,
    agents: list[str],
    num_agents: int,
    rng: jax.Array,
    n_env: int,
    n_steps: int,
    init_state,
    greedy_step,
    dict_to_agent_array,
) -> dict:
    """Stateful frozen (epsilon=0) greedy rollout for recurrent baselines (IPPO/MAPPO).

    The recurrent ``greedy_eval_rollout`` cannot carry a hidden state (or a RNN done
    reset signal), so this variant threads an opaque ``state`` (e.g. the RNN hidden
    state) through the rollout. ``greedy_step`` is responsible for any internal
    batching of the agent arrays.

    Args:
        env, agents, num_agents, rng, n_env, n_steps, dict_to_agent_array: as in
            :func:`greedy_eval_rollout`.
        init_state: initial value of the carried state (e.g. a zero RNN hidden state).
        greedy_step: callable ``(state, obs, avail, done) -> (new_state, actions)``
            where ``obs``/``avail`` have shape ``(n_env, num_agents, dim)``, ``done``
            has shape ``(n_env,)`` and is the *previous* step's global done (used as
            the RNN reset signal), and actions have shape ``(n_env, num_agents)``.
    """
    rng, rng_reset = jax.random.split(rng)
    obs_dict, env_state = jax.vmap(env.reset)(jax.random.split(rng_reset, n_env))
    obs = dict_to_agent_array(obs_dict, agents)
    done = jnp.zeros(n_env, dtype=jnp.bool_)

    def _eval_step(carry, _):
        env_state, obs, done, state, rng = carry
        rng, rng_step = jax.random.split(rng, 2)
        avail_dict = jax.vmap(env.get_avail_actions)(env_state.env_state)
        avail = dict_to_agent_array(avail_dict, agents).astype(jnp.float32)
        state, actions = greedy_step(state, obs, avail, done)
        env_act = {agent: actions[:, i] for i, agent in enumerate(agents)}
        step_rngs = jax.random.split(rng_step, n_env)
        next_obs_dict, env_state, _, dones_dict, infos = jax.vmap(env.step)(
            step_rngs, env_state, env_act
        )
        next_obs = dict_to_agent_array(next_obs_dict, agents)
        done = dones_dict["__all__"]
        return (env_state, next_obs, done, state, rng), infos

    _, infos = jax.lax.scan(
        _eval_step, (env_state, obs, done, init_state, rng), None, n_steps
    )
    return finished_episode_stats(infos, num_agents)

