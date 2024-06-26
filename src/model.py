from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import mctx
import pgx
from pgx.experimental import act_randomly


class Config(NamedTuple):
    env_id: pgx.EnvId
    seed: int = 0
    num_simulations: int = 1000
    batch_size: int = 1


HUMAN = 0
AI = 1

config = Config(env_id="connect_four")
env = pgx.make(config.env_id)
key = None
state = None
step_fn = None
_player_turn = None


def get_player_turn():
    return _player_turn


def policy_fn(legal_action_mask):
    """Return the logits of random policy. -Inf is set to illegal actions."""
    chex.assert_shape(legal_action_mask, (env.num_actions,))

    f_logits = legal_action_mask.astype(jnp.float32)
    logits = jnp.where(legal_action_mask, f_logits, jnp.finfo(f_logits.dtype).min)
    return logits


def value_fn(key, state):
    """Return the value based on random rollout."""
    chex.assert_rank(state.current_player, 0)

    def cond_fn(x):
        state, key = x
        return ~state.terminated

    def body_fn(x):
        state, key = x
        key, key_act, key_step = jax.random.split(key, 3)
        action = act_randomly(key_act, state.legal_action_mask)
        state = env.step(state, action, key_step)
        return (state, key)

    current_player = state.current_player
    state, _ = jax.lax.while_loop(cond_fn, body_fn, (state, key))
    return state.rewards[current_player]


def recurrent_fn(params, rng_key, action, state):
    del params
    current_player = state.current_player
    state = env.step(state, action)
    logits = policy_fn(state.legal_action_mask)
    value = value_fn(rng_key, state)
    reward = state.rewards[current_player]
    value = jax.lax.select(state.terminated, 0.0, value)
    discount = jax.lax.select(state.terminated, 0.0, -1.0)
    recurrent_fn_output = mctx.RecurrentFnOutput(
        reward=reward,
        discount=discount,
        prior_logits=logits,
        value=value,
    )
    return recurrent_fn_output, state


def run_mcts(key, state):
    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, config.batch_size)
    key, subkey = jax.random.split(key)

    root = mctx.RootFnOutput(
        prior_logits=jax.vmap(policy_fn)(state.legal_action_mask),
        value=jax.vmap(value_fn)(keys, state),
        embedding=state,
    )
    policy_output = mctx.muzero_policy(
        params=None,
        rng_key=subkey,
        root=root,
        invalid_actions=~state.legal_action_mask,
        recurrent_fn=jax.vmap(recurrent_fn, in_axes=(None, None, 0, 0)),
        num_simulations=config.num_simulations,
        max_depth=env.observation_shape[0]
        * env.observation_shape[1],  # 42 in connect four
        qtransform=partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),
        dirichlet_fraction=0.0,
    )
    return policy_output


def setup_jit():
    global config, key, state, step_fn, _player_turn
    _player_turn = HUMAN
    key = jax.random.PRNGKey(config.seed)
    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, config.batch_size)
    state = init_fn(keys)
    pgx.save_svg(state, "state.svg")


def human_move(move):
    global state, step_fn, _player_turn
    action = jnp.int32([move])
    state = step_fn(state, action)
    is_terminated = state.terminated.all()
    pgx.save_svg(state, "state.svg")
    _player_turn = 1 - _player_turn
    return is_terminated


def ai_move():
    global state, key, step_fn, _player_turn
    policy_output = jax.jit(run_mcts)(key, state)
    action = policy_output.action
    state = step_fn(state, action)
    is_terminated = state.terminated.all()
    pgx.save_svg(state, "state.svg")
    _player_turn = 1 - _player_turn
    return is_terminated
