"""A JAX implementation of batched MCTS."""
import functools
from typing import Any, NamedTuple, Optional, Tuple, TypeVar

import chex
import jax
import jax.numpy as jnp

from mctx._src import action_selection
from mctx._src import base
from mctx._src import tree as tree_lib
from jax.experimental.host_callback import call,id_print
Tree = tree_lib.Tree
T = TypeVar("T")

def search(
    params: base.Params,  #agent nn  #params to be forwarded to root and recurrent functions.
    rng_key: chex.PRNGKey, *, #random number generator state, the key is consumed.
    root: base.RootFnOutput,  #root`(prior::[B,A], value[B], embedding::env[B,...])` `RootFnOutput`. 
    recurrent_fn: base.RecurrentFn, #a callable to be called on the leaf nodes and unvisited
    #actions retrieved by the simulation step, which takes as args
    #`(params, rng_key, action, embedding)` and returns a `RecurrentFnOutput`
    #and the new state embedding. The `rng_key` argument is consumed.
    root_action_selection_fn: base.RootActionSelectionFn,  #function used to select an action at the root.
    interior_action_selection_fn: base.InteriorActionSelectionFn,# function used to select an action during simulation.
    num_simulations: int,  #32
    max_depth: Optional[int] = None,  #None #maximum search tree depth allowed during simulation, defined as
    #the number of edges from the root to a leaf node.
    invalid_actions: Optional[chex.Array] = None, #env.invalid_action()`[B, A]` a mask with invalid actions "at the root" invalid actions have ones, and valid actions have zeros.
    #   it's used at root node with seq halving
    extra_data: Any = None,  #gumbel distribution at policies.py  #extra data passed to `tree.extra_data`. Shape `[B, ...]`.
    loop_fn: base.LoopFn = jax.lax.fori_loop  #loop 32 times  #(It may be required to pass hk.fori_loop if using this function inside a Haiku module.)
  ) -> Tree: 
  """Performs a full search and returns sampled actions."""
  action_selection_fn = action_selection.switching_action_selection_wrapper(
      root_action_selection_fn=root_action_selection_fn,
      interior_action_selection_fn=interior_action_selection_fn
  )  #if depth == 0 root ,else interior

  #* Do simulation, expansion, and backward steps.
  batch_size = root.value.shape[0]  # len(B)
  batch_range = jnp.arange(batch_size) # 0 ~ B
  if max_depth is None:
    max_depth = num_simulations   #32
  if invalid_actions is None:
    invalid_actions = jnp.zeros_like(root.prior_logits)

  def body_fun(sim, loop_state):  #*for sim in num_simu:  #  0~num_simu = 0~32
    rng_key, tree = loop_state;    rng_key, simulate_key, expand_key = jax.random.split(rng_key, 3);
    # simulate is vmapped and expects batched rng keys.
    simulate_keys = jax.random.split(simulate_key, batch_size)
  #* rollout  root -> leaf
    parent_index, action = simulate( simulate_keys, tree, action_selection_fn, max_depth) 
    # A node first expanded on simulation `i`, will have node index `i`.
    # Node 0 corresponds to the root node.
    next_node_index = tree.children_index[batch_range, parent_index, action]   #-1   or same if it is terminated
    next_node_index = jnp.where(next_node_index == Tree.UNVISITED,  #* next_node_index = sim_id
                                sim + 1, next_node_index)
  #* expand
    tree = expand( params, expand_key, tree, recurrent_fn, parent_index,    #todo
        action, next_node_index)
  #* backup
    tree = backward(tree, next_node_index)
    loop_state = rng_key, tree
    return loop_state

  # Allocate all necessary storage.
  tree = instantiate_tree_from_root(root, num_simulations,
                                    root_invalid_actions=invalid_actions,
                                    extra_data=extra_data)
  _, tree = loop_fn(0, num_simulations, body_fun, (rng_key, tree))
  return tree

#*done
class _SimulationState(NamedTuple):
  """The state for the simulation while loop."""
  rng_key: chex.PRNGKey
  node_index: int
  action: int
  next_node_index: int
  depth: int
  is_continuing: bool

#*done
@functools.partial(jax.vmap, in_axes=[0, 0, None, None], out_axes=0)
def simulate(rng_key: chex.PRNGKey, #random number generator state, the key is consumed.
    tree: Tree, #_unbatched_ MCTS tree state.
    action_selection_fn: base.InteriorActionSelectionFn, # function used to select an action during simulation.
    max_depth: int # maximum search tree depth allowed during simulation.
  ) -> Tuple[chex.Array, chex.Array]:# (parent_index, action), where `parent_index` is the index of the
   # node reached at the end of the simulation, and the `action` is the action to
   # evaluate from the `parent_index`.  new_node = (a|Parent_node)
  """Traverses the tree until reaching an unvisited action or `max_depth`.  
  #*end if (root -> leaf || when max_depth is reached)  """
  def cond_fun(state):
    return state.is_continuing  #if not (reach_leafnode || depth > max_depth)

  def body_fun(state):
    # Preparing the next simulation state.
    node_index = state.next_node_index
    rng_key, action_selection_key = jax.random.split(state.rng_key)
    action = action_selection_fn(action_selection_key, tree, node_index,
                                 state.depth)
    next_node_index = tree.children_index[node_index, action]
    # The returned action will be visited.
    depth = state.depth + 1
    is_before_depth_cutoff = depth < max_depth
    is_visited = next_node_index != Tree.UNVISITED  #-1
    is_continuing = jnp.logical_and(is_visited, is_before_depth_cutoff)
    return _SimulationState( rng_key=rng_key,
        node_index=node_index,  action=action,  #int
        next_node_index=next_node_index, depth=depth,
        is_continuing=is_continuing)

  node_index = jnp.array(Tree.ROOT_INDEX, dtype=jnp.int32)
  depth = jnp.zeros((), dtype=tree.children_prior_logits.dtype)  #0
  initial_state = _SimulationState( rng_key=rng_key,
      node_index=tree.NO_PARENT,  action=tree.NO_PARENT, #-1
      next_node_index=node_index, depth=depth,
      is_continuing=jnp.array(True))
  end_state = jax.lax.while_loop(cond_fun, body_fun, initial_state)  #while(state.is_continuing)
  #* end_state.next_node_index   is unvisted     end_state.action   =  action_from_parent(end_state.next_node_index)
  # Returning a node with a selected action.
  # The action can be already visited, if the max_depth is reached.
  return end_state.node_index, end_state.action
#*done
def expand(
    params: chex.Array,  #agent_nn #params to be forwarded to recurrent function.
    rng_key: chex.PRNGKey, #random number generator state.
    tree: Tree[T], # the MCTS tree state to update.
    recurrent_fn: base.RecurrentFn,  #* env.step  get r,v,prior
    # a callable to be called on the leaf nodes and unvisited
    # actions retrieved by the simulation step, which takes as args
    #`(params, rng_key, action, embedding=env)` and returns a `RecurrentFnOutput`   
    #  and the new state embedding. The `rng_key` argument is consumed.
    # #todo   this  is the power of muzero , you give  a,S  and it can generate p,v directly  (without seeing S'=(a|S))
    parent_index: chex.Array,  #`[B]` parent(new_node) #the index of the parent node, 
    # from which the action will be expanded.
    action: chex.Array,  #`[B]` action from parent(new_node)  # the action to expand
    next_node_index: chex.Array  #`[B]`new_node  #the index of the newly expanded node. 
    #*This can be the index of an existing node, if `max_depth` is reached.
  ) -> Tree[T]:  # updated MCTS tree state.
  """Create and evaluate child nodes from given nodes and unvisited actions.  """
  batch_size = tree_lib.infer_batch_size(tree)
  batch_range = jnp.arange(batch_size);  chex.assert_shape([parent_index, action, next_node_index], (batch_size,))

  # Retrieve states for nodes to be evaluated.
  embedding = jax.tree_util.tree_map(  #parent_node = parent(new_node)
      lambda x: x[batch_range, parent_index], tree.embeddings)  #B,N=parent_idx,env ->  B,env

  # Evaluate and create a new node.
  step, embedding = recurrent_fn(params, rng_key, action, embedding)  #* new_node's state
  #* step = RecurrentFnOutput(reward,discount,prior,value)
  chex.assert_shape(step.prior_logits, [batch_size, tree.num_actions])
  chex.assert_shape(step.reward, [batch_size])
  chex.assert_shape(step.discount, [batch_size])
  chex.assert_shape(step.value, [batch_size])
  tree = update_tree_node(
      tree, next_node_index, step.prior_logits, step.value, embedding)

  # Return updated tree topology.
  return tree.replace(
      children_index=batch_update(
          tree.children_index, next_node_index, parent_index, action), #children_index[parent_id,action] = next_node_index
      children_rewards=batch_update(
          tree.children_rewards, step.reward, parent_index, action),
      children_discounts=batch_update(
          tree.children_discounts, step.discount, parent_index, action),
      parents=batch_update(tree.parents, parent_index, next_node_index), #parents(next_node_id) = parent_id
      action_from_parent=batch_update(    #action_from_parent[next_node_id] = action
          tree.action_from_parent, action, next_node_index))

#* done
@jax.vmap
def backward(
    tree: Tree[T], #the MCTS tree state to update, without the batch size.
    leaf_index: chex.Numeric #newly generated index # the node index from which to do the backward.  
  ) -> Tree[T]:#  Updated MCTS tree state.
  """Goes up and updates the tree until all nodes reached the root.  """

  def cond_fun(loop_state):
    _, _, index = loop_state
    return index != Tree.ROOT_INDEX

  def body_fun(loop_state):
    # Here we update the value of our parent, so we start by reversing.
    tree, leaf_value, index = loop_state  # index from leaf -> root
    parent = tree.parents[index]
    count = tree.node_visits[parent]
    action = tree.action_from_parent[index]
    reward = tree.children_rewards[parent, action]
    leaf_value = reward + tree.children_discounts[parent, action] * leaf_value  # discount = -1  #2 player game
    parent_value = (
        tree.node_values[parent] * count + leaf_value) / (count + 1.0)   #  Q*N+v / (N+1)
    children_values = tree.node_values[index]
    children_counts = tree.children_visits[parent, action] + 1

    tree = tree.replace(
        node_values=update(tree.node_values, parent_value, parent),  #
        node_visits=update(tree.node_visits, count + 1, parent),
        children_values=update(
            tree.children_values, children_values, parent, action),
        children_visits=update(
            tree.children_visits, children_counts, parent, action))

    return tree, leaf_value, parent  #keep going up to root

  leaf_index = jnp.asarray(leaf_index, dtype=jnp.int32)  #newly generated index
  loop_state = (tree, tree.node_values[leaf_index], leaf_index)  # tree, v at leaf, leaf_idx
  tree, _, _ = jax.lax.while_loop(cond_fun, body_fun, loop_state) #while (index!= root_index)

  return tree

#*done
# Utility function to set the values of certain indices to prescribed values.
# This is vmapped to operate seamlessly on batches.
def update(x, vals, *indices):
  return x.at[indices].set(vals)


batch_update = jax.vmap(update)

#* add new node and its value, prior to the tree
def update_tree_node(
    tree: Tree[T], #`Tree` to whose node is to be updated.
    node_index: chex.Array,#`[B]`.the index of the expanded node. #newly created node
    prior_logits: chex.Array, #`[B, A]`the prior logits to fill in for the new node
    value: chex.Array, #`[B]`. the value to fill in for the new node
    embedding: chex.Array #`[B,N,env]`. the state embeddings for the node. Shape 
  ) -> Tree[T]:  #The new tree with updated nodes.
  """Updates the tree at node index."""
  batch_size = tree_lib.infer_batch_size(tree)   #*@ tree.py
  batch_range = jnp.arange(batch_size);  chex.assert_shape(prior_logits, (batch_size, tree.num_actions))

  # When using max_depth, a leaf can be expanded multiple times.
  new_visit = tree.node_visits[batch_range, node_index] + 1
  updates = dict(  # pylint: disable=use-dict-literal
      children_prior_logits=batch_update(
          tree.children_prior_logits, prior_logits, node_index),
      raw_values=batch_update(
          tree.raw_values, value, node_index),   #!! node value == raw value ??
      node_values=batch_update(
          tree.node_values, value, node_index),
      node_visits=batch_update(
          tree.node_visits, new_visit, node_index),
      embeddings=jax.tree_util.tree_map(
          lambda t, s: batch_update(t, s, node_index),
          tree.embeddings, embedding))

  return tree.replace(**updates)  #** means kwargs https://stackoverflow.com/questions/36901/what-does-double-star-asterisk-and-star-asterisk-do-for-parameters

#*done
def instantiate_tree_from_root(
    root: base.RootFnOutput,#root`(prior::[B,A], value[B], embedding::env[B,...])` `RootFnOutput`. 
    num_simulations: int,  #32
    root_invalid_actions: chex.Array,  # `[B, A]`
    extra_data: Any  #gumbel distribution
  ) -> Tree:
  """Initializes tree state at search root."""
  chex.assert_rank(root.prior_logits, 2)
  batch_size, num_actions = root.prior_logits.shape;  chex.assert_shape(root.value, [batch_size])
  num_nodes = num_simulations + 1;  data_dtype = root.value.dtype
  batch_node = (batch_size, num_nodes)    #[B,N] = [Batch_size, Tree_size]
  batch_node_action = (batch_size, num_nodes, num_actions)  #[B,N,A]

  def _zeros(x):       #go_env {board,recent_boards,...DSU}      # skip Batch   #[B,env]
    # call(lambda x: print(f"aaa: {x.shape}, {x.dtype}, {x}"), x)   #** print this is very fun
    #                             remove batch
    return jnp.zeros(batch_node + x.shape[1:], dtype=x.dtype)
  #root.embedding
  # Create a new empty tree state and fill its root.
  # id_print(len(root.embedding))
  # id_print(root.embedding[0].shape)
  tree = Tree(
      node_visits=jnp.zeros(batch_node, dtype=jnp.int32),
      raw_values=jnp.zeros(batch_node, dtype=data_dtype),
      node_values=jnp.zeros(batch_node, dtype=data_dtype),
      parents=jnp.full(batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      action_from_parent=jnp.full(   # fill -1
          batch_node, Tree.NO_PARENT, dtype=jnp.int32),
      children_index=jnp.full(  #fill -1
          batch_node_action, Tree.UNVISITED, dtype=jnp.int32),
      children_prior_logits=jnp.zeros(
          batch_node_action, dtype=root.prior_logits.dtype),
      children_values=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_visits=jnp.zeros(batch_node_action, dtype=jnp.int32),
      children_rewards=jnp.zeros(batch_node_action, dtype=data_dtype),
      children_discounts=jnp.zeros(batch_node_action, dtype=data_dtype),
      embeddings=jax.tree_util.tree_map(_zeros, root.embedding),   #** B,N, env
      root_invalid_actions=root_invalid_actions,
      extra_data=extra_data)

  root_index = jnp.full([batch_size], Tree.ROOT_INDEX)  #0
  tree = update_tree_node(
      tree, root_index, root.prior_logits, root.value, root.embedding)
  return tree
