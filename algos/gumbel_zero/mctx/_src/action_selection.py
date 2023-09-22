"""A collection of action selection functions."""
from typing import Optional, TypeVar

import chex
import jax
import jax.numpy as jnp
from jax.experimental.host_callback import id_print,call,id_tap
from mctx._src import base
from mctx._src import qtransforms
from mctx._src import seq_halving
from mctx._src import tree as tree_lib

#** called in search.py       
# action_selection_fn(action_selection_key, tree, node_index,state.depth)
def switching_action_selection_wrapper(
    root_action_selection_fn: base.RootActionSelectionFn,
    interior_action_selection_fn: base.InteriorActionSelectionFn
) -> base.InteriorActionSelectionFn:
  """Wraps root and interior action selection fns in a conditional statement."""

  def switching_action_selection_fn(
      rng_key: chex.PRNGKey,
      tree: tree_lib.Tree,
      node_index: base.NodeIndices,
      depth: base.Depth) -> chex.Array:
    return jax.lax.cond(
        depth == 0,
        lambda x: root_action_selection_fn(*x[:3]),#if depth == 0 -> root_action(rng,tree,node_idx)
        lambda x: interior_action_selection_fn(*x),#else interior_action(rng_key, tree, node_index, depth) 
        (rng_key, tree, node_index, depth)
      )

  return switching_action_selection_fn


@chex.dataclass(frozen=True)
class GumbelMuZeroExtraData:
  """Extra data for Gumbel MuZero search."""
  root_gumbel: chex.Array
GumbelMuZeroExtraDataType = TypeVar(  # pylint: disable=invalid-name
    "GumbelMuZeroExtraDataType", bound=GumbelMuZeroExtraData)
#done
def gumbel_muzero_root_action_selection(
    rng_key: chex.PRNGKey, #random number generator state.
    tree: tree_lib.Tree[GumbelMuZeroExtraDataType], #_unbatched_ MCTS tree state.
    node_index: chex.Numeric,   #scalar index of the node from which to take an action
    *,
    num_simulations: chex.Numeric,   #ex. 32  simulation budget.
    max_num_considered_actions: chex.Numeric,   #16   the number of actions sampled without replacement.
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value, # a monotonic transformation for the Q-values.
) -> chex.Array: # Returns: action: the action selected from the given node.
  """  root_action(rng,tree,node_idx,_,num_simu=32,max_num_a=16,qtrans_complete)  called in search.py action_selection_fn  and policies.py 
  Returns the action selected by Sequential Halving with Gumbel.
  
                   # sample with replacement: 1/10 * 1/10  -> sample without replacement: 1/10 * 1/9
  Initially, we sample `max_num_considered_actions` actions #*without replacement.   (full exploration, won't repeat)
  From these, the actions with the highest `gumbel + logits + qvalues` are
  visited first.
  seq halving is well-suited for pure exploration tasks. (compare to regret-minimizing algorithm like UCB)
  """
  del rng_key
  chex.assert_shape([node_index], ())
  visit_counts = tree.children_visits[node_index]  #visit_counts::[B,A]
  prior_logits = tree.children_prior_logits[node_index]  #prior::[B,A]
  chex.assert_equal_shape([visit_counts, prior_logits])
  completed_qvalues = qtransform(tree, node_index)  #* [B,A]   value of node idx

  #* some setup for seq halving 
  table = jnp.array(seq_halving.get_table_of_considered_visits(
      max_num_considered_actions, num_simulations))   # just a table(cheat sheet) for seq halving 
  """
  2  [ 0 0 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 12 12 13 13 14 14 15 15 ]
  3  [ 0 0 0 1 1 1 2 2 2 3 3 3 4 4 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 12 12 13 ]
  4  [ 0 0 0 0 1 1 1 1 2 2 2 2 3 3 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 ]
  5  [ 0 0 0 0 0 1 1 1 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 12 12 ]
  6  [ 0 0 0 0 0 0 1 1 1 2 2 2 3 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 12 ]
  7  [ 0 0 0 0 0 0 0 1 1 1 2 2 2 3 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 11 11 ]
  8  [ 0 0 0 0 0 0 0 0 1 1 1 1 2 2 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 10 ]
  9  [ 0 0 0 0 0 0 0 0 0 1 1 1 1 2 2 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 ]
  10 [ 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 10 ]
  11 [ 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 2 2 3 3 4 4 5 5 6 6 7 7 8 8 9 9 ]
  12 [ 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 3 3 3 4 4 5 5 6 6 7 7 ]
  13 [ 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 2 2 2 3 3 3 4 4 5 5 6 6 7 ]
  14 [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 2 2 2 3 3 3 4 4 5 5 6 ]
  15 [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 2 2 2 3 3 3 4 4 5 5 ]
  16 [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 3 3 3 3 ]
  """
  
  num_valid_actions = jnp.sum(   
      1 - tree.root_invalid_actions, axis=-1).astype(jnp.int32)
  num_considered = jnp.minimum(
      max_num_considered_actions, num_valid_actions)   #16
  chex.assert_shape(num_considered, ())

  # At the root, the simulation_index is equal to the sum of visit counts.
  simulation_index = jnp.sum(visit_counts, -1)  # 0 ~ simu num,   ith run of rollout
  chex.assert_shape(simulation_index, ())
  considered_visit = table[num_considered, simulation_index]   # num_considered = 16   [0]8+[1]4+[2]4+[3]2
  #   0  or 1  or 2 or 3 :: int  
  chex.assert_shape(considered_visit, ())
  gumbel = tree.extra_data.root_gumbel   #jax.random.gumbel  @ policies.py
  to_argmax = seq_halving.score_considered(      #[B,A]
      considered_visit, gumbel, prior_logits, completed_qvalues,
      visit_counts)

  # Masking the invalid actions at the root.
  return masked_argmax(to_argmax, tree.root_invalid_actions)  #int  #selected deterministic action
#done
def gumbel_muzero_interior_action_selection(
    rng_key: chex.PRNGKey, # random number generator state.
    tree: tree_lib.Tree,   #_unbatched_ MCTS tree state.
    node_index: chex.Numeric,  #::int scalar index of the node from which to take an action.
    depth: chex.Numeric,  #the scalar depth of the current node. The root has depth zero.
    *,
    qtransform: base.QTransform = qtransforms.qtransform_completed_by_mix_value,
    # function to obtain completed Q-values for a node.
) -> chex.Array:  #Returns: action: the action selected from the given node.
  """
  interior_action(rng_key, tree, node_index, depth)   called in search.py action_selection_fn
  Selects the action with a deterministic action selection.
  #* section 5 in the paper
  The action is selected based on the visit counts to produce visitation
  frequencies similar to softmax(prior_logits + qvalues).
  """
  del rng_key, depth;  chex.assert_shape([node_index], ())
  visit_counts = tree.children_visits[node_index]        #  (A)
  prior_logits = tree.children_prior_logits[node_index]  #  (A)
  chex.assert_equal_shape([visit_counts, prior_logits])
  completed_qvalues = qtransform(tree, node_index)    #(A)
  # The `prior_logits + completed_qvalues` provide an improved policy,
  # because the missing qvalues are replaced by v_{prior_logits}(node).
  # print(prior_logits)  #  = bs
  
  '''
  # completed_qvalues = jnp.where(completed_qvalues>0,100*completed_qvalues,1)
  # probs=jax.nn.softmax(prior_logits*completed_qvalues)    this line is equivalent to things we did below
  a.k.a
  # probs=jax.nn.softmax(prior_logits+100*completed_qvalues),
  '''
  
  # id_print(prior_logits) #bs,  A 
  # id_print(completed_qvalues)   #**  notice that completed q value is very sparse mostly 0.0  
                                #** because most of the time your action dont get value
                                #**  ex. action 5 [0,0,0,0,12,0]
                                #**               [0,0,0,0,0,0]   at most of the time
  to_argmax = _prepare_argmax_input(
    #                                     c1
      probs=jax.nn.softmax(prior_logits+completed_qvalues),    ##prior is super inportant, without it the performance drop a lot  #*  it's similar to dueling Q network
                                                                    ##  prior+0*complete >>  0*prior+complete  
                                                              ## it's because  Q is sparse (doen't has much contribution )
                                                              ## so the influential of Q　is pretty tiny, that we can almost forget this hyperparemter
                                                              ##  add Q into consideration only slighty improve a little performance
                                                              ## but the main contributor is still on prior_logits & visit_count (exploration? dont't visit same node too much)
                                                              ##                                      (where to go)
                                                              #   you can't use Q to dicide where to go (because it is too sparse)
                                                              #completed q value is completed by value of parent, should be value of child but it is computation prohibitive                
      visit_counts=visit_counts)  
  chex.assert_rank(to_argmax, 1)   
                                   

  """
             exploit     explore      (if N(a) is too large, point will get down, forcing it to explore)
  arg max_a {prior(a) - N(a)/(1+N_tot)  }
  """
  return jnp.argmax(to_argmax, axis=-1)   #* it is deterministic, what if I use sampling?? (may lead to terrible decision at critical point)

#done
def masked_argmax(
    to_argmax: chex.Array,
    invalid_actions: Optional[chex.Array]) -> chex.Array:
  """Returns a valid action with the highest `to_argmax`.  (A)->(1,)"""
  if invalid_actions is not None:
    chex.assert_equal_shape([to_argmax, invalid_actions])
    # The usage of the -inf inside the argmax does not lead to NaN.
    # Do not use -inf inside softmax, logsoftmax or cross-entropy.
    to_argmax = jnp.where(invalid_actions, -jnp.inf, to_argmax)
  # If all actions are invalid, the argmax returns action 0.
  return jnp.argmax(to_argmax, axis=-1)


def _prepare_argmax_input(
    probs,  #(A,) a policy or an improved policy. Shape `[num_actions]`.
    visit_counts #(A,) the existing visit counts
  ):  #Returns: (A,) The input to an argmax. Shape
  """Prepares the input for the deterministic selection.
  When calling argmax(_prepare_argmax_input(...)) multiple times
  with updated visit_counts, the produced visitation frequencies will
  approximate the probs.
  For the derivation, see Section 5 "Planning at non-root nodes" in
  "Policy improvement by planning with Gumbel":https://openreview.net/forum?id=bERaNdoegnO
    arg max_a {prior(a) - N(a)/(1+N_tot)  }
  """
  chex.assert_equal_shape([probs, visit_counts])
  to_argmax = probs - 0*visit_counts / (  #!!  it seems that visit_counts doesn't matter   
      1 + jnp.sum(visit_counts, keepdims=True, axis=-1))      #? so we don't need exploration at interior node ???  
  return to_argmax
