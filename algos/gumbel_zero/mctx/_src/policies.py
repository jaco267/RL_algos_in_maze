"""Search policies."""
import functools
from typing import Optional, Tuple

import chex
import jax
import jax.numpy as jnp

from mctx._src import action_selection
from mctx._src import base
from mctx._src import qtransforms
from mctx._src import search
from mctx._src import seq_halving

from jax.experimental.host_callback import call,id_print  #for printing jit compiled fn  #https://github.com/google/jax/issues/196


def gumbel_muzero_policy(
    #`B`: batch dimension.  `A`: num_actions
    params: base.Params, # agent_NN  params to be forwarded to root and recurrent functions
    rng_key: chex.PRNGKey, # random number generator state, the key is consumed.
    root: base.RootFnOutput,   #`(prior, value, embedding)` `RootFnOutput`. shapes are `([B, A], [B], [B, ...])`
    recurrent_fn: base.RecurrentFn,   # a fn be called on the leaf nodes and unvisited
    # actions retrieved by the simulation step, which takes as args`(params, rng_key, action, embedding)`
    # and returns a `RecurrentFnOutput` and the new state embedding. The `rng_key` argument is consumed.
    num_simulations: int,   #32
    invalid_actions: Optional[chex.Array] = None,   #a mask with invalid actions. Invalid actions have ones, valid actions have zeros in the mask. Shape `[B, num_actions]`.
    max_depth: Optional[int] = None,   #maximum search tree depth allowed during simulation.
    loop_fn: base.LoopFn = jax.lax.fori_loop,   #Function used to run the simulations. It may be required to pass hk.fori_loop if using this function inside a Haiku module.
    *,
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value,   #function to obtain completed Q-values for "a node".
    max_num_considered_actions: int = 16,   #the maximum number of actions expanded at the root node.
    # A smaller number of actions will be expanded if the number of valid actions is smaller.
    gumbel_scale: chex.Numeric = 1.,   #scale for the Gumbel noise. Evalution on perfect-information games can use gumbel_scale=0.0.
) -> base.PolicyOutput[action_selection.GumbelMuZeroExtraData]:
  """Runs Gumbel MuZero search and returns the `PolicyOutput`.
  Full Gumbel MuZero policy from "Policy improvement by planning with Gumbel". https://openreview.net/forum?id=bERaNdoegnO

  At the root node, actions are selected by Sequential Halving with Gumbel. 
  At non-root nodes (aka interior nodes), actions are selected by 
  Full Gumbel MuZero deterministic action selection.

  Returns:
    `PolicyOutput` (action(proposed by MCTS), action_weights , the used search tree )
  """
  # Masking invalid actions.
  root = root.replace( #https://docs.python.org/3.7/library/dataclasses.html#dataclasses.replace
      prior_logits=_mask_invalid_actions(root.prior_logits, invalid_actions))

  # Generating Gumbel.
  rng_key, gumbel_rng = jax.random.split(rng_key)
  gumbel = gumbel_scale * jax.random.gumbel(
      gumbel_rng, shape=root.prior_logits.shape, dtype=root.prior_logits.dtype)
#   print('gumbel_shape',root.prior_logits.shape)
  #                     [B,A]
  #*-----start simulation----
  # Searching.    #?32 times??
  extra_data = action_selection.GumbelMuZeroExtraData(root_gumbel=gumbel)  #just a dataclass
  search_tree = search.search(
      params=params,    rng_key=rng_key,
      root=root,      recurrent_fn=recurrent_fn,
      root_action_selection_fn=functools.partial(
          action_selection.gumbel_muzero_root_action_selection,
          num_simulations=num_simulations,
          max_num_considered_actions=max_num_considered_actions,
          qtransform=qtransform,
      ),
      interior_action_selection_fn=functools.partial(
          action_selection.gumbel_muzero_interior_action_selection,
          qtransform=qtransform,
      ),
      num_simulations=num_simulations,  max_depth=max_depth,
      invalid_actions=invalid_actions,  extra_data=extra_data,  #gumbel noise
      loop_fn=loop_fn)
  summary = search_tree.summary()
  #***  ----select determinstic best action to take in real env after 32 mcts simulation(using resulting Q )---
  # Acting with the "best action"(highest `gumbel + logits + q`.) from the most visited actions.
  # Inside the minibatch, the considered_visit can be different on states with a smaller number of valid actions.
  considered_visit = jnp.max(summary.visit_counts, axis=-1, keepdims=True)  #(visit_counts of root node child [bs,A=82])  self.children_visits[:, Tree.ROOT_INDEX] #at tree.py
  # call(lambda x: print(f"aaa: {x}"), considered_visit)   #ex. tot 32 sim   child visit of node would be ex. [4,4,0,2,8,0,15,0,,0]  #sum <=32
  
  # The completed_qvalues include imputed values for unvisited actions.
  completed_qvalues = jax.vmap(qtransform, in_axes=[0, None])(    # oneNode, vmap for each batch, so it only pass shape(A) into the func
      search_tree, search_tree.ROOT_INDEX)  # [B,N,A] -> [vmap(B),N(Root_index),A] -> [A]
  
  # select the "best action" after 32 sim (by seq halving), and use this best action in the real world
  to_argmax = seq_halving.score_considered(considered_visit, gumbel, root.prior_logits,
      completed_qvalues,  summary.visit_counts)  #[B,A]    #elimate action to -inf  that is not a top-k candidate  (top-k candidate all have the same highest visit count)
  action = action_selection.masked_argmax(to_argmax, invalid_actions)   #todo   wait  isn't it better to put this in front of score_considered?  or the top k might all be invalid actions??
  # call(lambda x: print(f"aaa: {x}"), action)  action::int (bs,) = selected action that will take in real env
  # Producing action_weights usable to train the policy network.
  completed_search_logits = _mask_invalid_actions(
      root.prior_logits + completed_qvalues, invalid_actions)  #?? wait  why is this "+"??  so it is similar to posterior = prior + q??
                                                            #todo I think the variance can be further lower if we can change this heuristic to regularized policy optim
                                                            #* this formula is dicussed at 4. LEARNING AN IMPROVED POLICY
  action_weights = jax.nn.softmax(completed_search_logits)
  return base.PolicyOutput(
      action=action,
      action_weights=action_weights,
      search_tree=search_tree)

#  Stochastic MuZero paper: (https://openreview.net/forum?id=X6D9bAHhBQ1).
#done
def _mask_invalid_actions(logits, invalid_actions):
  """Returns logits with zero mass to invalid actions."""
  if invalid_actions is None:
    return logits
  chex.assert_equal_shape([logits, invalid_actions])
  logits = logits - jnp.max(logits, axis=-1, keepdims=True)
  # At the end of an episode, all actions can be invalid. A softmax would then
  # produce NaNs, if using -inf for the logits. We avoid the NaNs by using
  # a finite `min_logit` for the invalid actions.
  min_logit = jnp.finfo(logits.dtype).min
  return jnp.where(invalid_actions, min_logit, logits)


