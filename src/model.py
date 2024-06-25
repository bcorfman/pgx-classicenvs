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


config = None
env = None
key = None
state = None
step_fn = None


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
    global config, env, key, state, step_fn
    config = Config(env_id="connect_four")
    pgx.make(config.env_id)
    assert config.batch_size == 1
    key = jax.random.PRNGKey(config.seed)
    init_fn = jax.jit(jax.vmap(env.init))
    step_fn = jax.jit(jax.vmap(env.step))

    key, subkey = jax.random.split(key)
    keys = jax.random.split(subkey, config.batch_size)
    state = init_fn(keys)
    pgx.save_svg(state, "state.svg")


def human_move(self, move):
    pgx.save_svg(self.state, "state.svg")
    if self.state.terminated.all():
        is_terminated = True
    else:
        action = jnp.int32([move])
        self.state = self.step_fn(self.state, action)
        is_terminated = False
    return is_terminated


def ai_move(self):
    pgx.save_svg(self.state, "state.svg")
    if self.state.terminated.all():
        is_terminated = True
    else:
        policy_output = jax.jit(self.run_mcts)(self.key, self.state)
        action = policy_output.action
        self.state = self.step_fn(self.state, action)
        is_terminated = False
    return is_terminated
