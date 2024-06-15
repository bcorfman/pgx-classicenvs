import os
import tkinter as tk
from functools import partial
from typing import NamedTuple

import chex
import jax
import jax.numpy as jnp
import mctx
import pgx
import solara
from pgx.experimental import act_randomly


class Config(NamedTuple):
    env_id: pgx.EnvId
    seed: int = 0
    num_simulations: int = 1000
    batch_size: int = 1


class RLGame:
    def __init__(self):
        self.config = Config(env_id=id)
        self.env = pgx.make(self.config.env_id)

    def policy_fn(self, legal_action_mask):
        """Return the logits of random policy. -Inf is set to illegal actions."""
        chex.assert_shape(legal_action_mask, (self.env.num_actions,))

        f_logits = legal_action_mask.astype(jnp.float32)
        logits = jnp.where(legal_action_mask, f_logits, jnp.finfo(f_logits.dtype).min)
        return logits

    def value_fn(self, key, state):
        """Return the value based on random rollout."""
        chex.assert_rank(state.current_player, 0)

        def cond_fn(x):
            state, key = x
            return ~state.terminated

        def body_fn(x):
            state, key = x
            key, key_act, key_step = jax.random.split(key, 3)
            action = act_randomly(key_act, state.legal_action_mask)
            state = self.env.step(state, action, key_step)
            return (state, key)

        current_player = state.current_player
        state, _ = jax.lax.while_loop(cond_fn, body_fn, (state, key))
        return state.rewards[current_player]

    def recurrent_fn(self, params, rng_key, action, state):
        del params
        current_player = state.current_player
        state = self.env.step(state, action)
        logits = self.policy(state.legal_action_mask)
        value = self.value(rng_key, state)
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

    def run_mcts(self, key, state):
        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, self.config.batch_size)
        key, subkey = jax.random.split(key)

        root = mctx.RootFnOutput(
            prior_logits=jax.vmap(self.policy_fn)(state.legal_action_mask),
            value=jax.vmap(self.value)(keys, state),
            embedding=state,
        )
        policy_output = mctx.muzero_policy(
            params=None,
            rng_key=subkey,
            root=root,
            invalid_actions=~state.legal_action_mask,
            recurrent_fn=jax.vmap(self.recurrent_fn, in_axes=(None, None, 0, 0)),
            num_simulations=self.config.num_simulations,
            max_depth=self.env.observation_shape[0]
            * self.env.observation_shape[1],  # 42 in connect four
            qtransform=partial(mctx.qtransform_by_min_max, min_value=-1, max_value=1),
            dirichlet_fraction=0.0,
        )
        return policy_output

    def vs_human(self, is_human_first=True):
        assert self.config.batch_size == 1
        key = jax.random.PRNGKey(self.config.seed)
        init_fn = jax.jit(jax.vmap(self.env.init))
        step_fn = jax.jit(jax.vmap(self.env.step))

        key, subkey = jax.random.split(key)
        keys = jax.random.split(subkey, self.config.batch_size)
        state: pgx.State = init_fn(keys)

        is_human_turn = is_human_first
        while True:
            pgx.save_svg(state, "state.svg")
            if state.terminated.all():
                break
            if is_human_turn:
                action = int(input("Your action: "))
                action = jnp.int32([action])
            else:
                policy_output = jax.jit(self.run_mcts)(key, state)
                action = policy_output.action
            state = step_fn(state, action)
            is_human_turn = not is_human_turn
            yield is_human_turn


def play_game(id):
    config = Config(env_id=id)
