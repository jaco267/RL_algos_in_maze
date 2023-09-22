"""A data structure used to hold / inspect search data for a batch of inputs."""

from __future__ import annotations
from typing import Any, ClassVar, Generic, TypeVar

import chex
import jax
import jax.numpy as jnp


T = TypeVar("T")


@chex.dataclass(frozen=True)
class Tree(Generic[T]):  #** tree is created at @search.py instantiate_tree_from_root()
  """State of a search tree.
  `Tree` dataclass is used to hold and inspect search data for a batch of
  inputs. 
  `B`: batch dim, 
  `N`: number of nodes in the tree, 
  `A` : number of discrete actions. 
  """
  node_visits: chex.Array  # [B, N]  the visit counts for each node.  #*  N(s)
  raw_values: chex.Array  #? [B, N]   the raw value for each node.     #??  v_pi estimated by NN
  node_values: chex.Array  #? [B, N]  the cumulative search value for each node.  #?? r + gamma*v_pi ??  see backward in search.py
  parents: chex.Array  # [B, N]  the node index for the parents for each node. #*parent(s)
  action_from_parent: chex.Array  # [B, N]  action to take from the parent to reach each node.  #* S_ = ("a"|S)
  children_index: chex.Array  # [B, N, A]  node index of the children for each action.  #* "S_" = (a|S)
  children_prior_logits: chex.Array  # [B, N, A]  the action prior logits of each node.  #* p(a|S)
  children_visits: chex.Array  # [B, N, A]  visit counts for children for each action. #* N(a|s)
  children_rewards: chex.Array  # [B, N, A] immediate reward for each action. # r(a|s)
  children_discounts: chex.Array  # [B, N, A] discount between the `children_rewards` and the `children_values`  #* gamma
  children_values: chex.Array  # [B, N, A]the value of the next node after the action.  #* V(a|s)
  embeddings: Any  # [B, N, env]   state embeddings of each node.   #* S  = env
  root_invalid_actions: chex.Array  # [B, A] a mask with invalid actions #*at the root.
  #   In the mask, invalid actions have ones, and valid actions have zeros.
  extra_data: T  # [B, ...] gumbel distri   #  extra data passed to the search.

  # The following attributes are class variables (and should not be set on
  # Tree instances).
  ROOT_INDEX: ClassVar[int] = 0
  NO_PARENT: ClassVar[int] = -1
  UNVISITED: ClassVar[int] = -1

  @property
  def num_actions(self):   # A
    return self.children_index.shape[-1]

  @property
  def num_simulations(self):  #N
    return self.node_visits.shape[-1] - 1

  def qvalues(self, indices):   # called below
    """Compute q-values for any node indices in the tree."""
    if jnp.asarray(indices).shape:   #bs,indice
      return jax.vmap(_unbatched_qvalues)(self, indices)  #qvaules = r   - 1 * children_value
    else:
      return _unbatched_qvalues(self, indices)

  def summary(self) -> SearchSummary:
    """Extract summary statistics for the root node."""
    # Get state and action values for the root nodes.
    chex.assert_rank(self.node_values, 2)
    value = self.node_values[:, Tree.ROOT_INDEX]  # `B`
    batch_size, = value.shape
    root_indices = jnp.full((batch_size,), Tree.ROOT_INDEX)  # [0]*bs
    qvalues = self.qvalues(root_indices)
    # Extract visit counts and induced probabilities for the root nodes.
    visit_counts = self.children_visits[:, Tree.ROOT_INDEX].astype(value.dtype)  # [B,A]
    total_counts = jnp.sum(visit_counts, axis=-1, keepdims=True)  # B
    visit_probs = visit_counts / jnp.maximum(total_counts, 1)   #Q
    visit_probs = jnp.where(total_counts > 0, visit_probs, 1 / self.num_actions)
    # Return relevant stats.
    return SearchSummary(  # pytype: disable=wrong-arg-types  # numpy-scalars
        visit_counts=visit_counts,
        visit_probs=visit_probs,
        value=value,
        qvalues=qvalues)


def infer_batch_size(tree: Tree) -> int:
  """Recovers batch size from `Tree` data structure."""
  if tree.node_values.ndim != 2:
    raise ValueError("Input tree is not batched.")
  chex.assert_equal_shape_prefix(jax.tree_util.tree_leaves(tree), 1)
  return tree.node_values.shape[0]


# A number of aggregate statistics and predictions are extracted from the
# search data and returned to the user for further processing.
@chex.dataclass(frozen=True)
class SearchSummary:
  """Stats from MCTS search."""
  visit_counts: chex.Array
  visit_probs: chex.Array
  value: chex.Array
  qvalues: chex.Array


def _unbatched_qvalues(tree: Tree, index: int) -> int:
  chex.assert_rank(tree.children_discounts, 2)
  return (  # pytype: disable=bad-return-type  # numpy-scalars
      tree.children_rewards[index]
      + tree.children_discounts[index] * tree.children_values[index]
  )
