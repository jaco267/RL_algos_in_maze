"""Functions for Sequential Halving."""

import math

import chex
import jax.numpy as jnp

from jax.experimental.host_callback import call,id_tap,id_print  #for printing jit compiled fn  #https://github.com/google/jax/issues/196

"""
 it can be used to select at root node   select deterministic after simulate @policies.gumbel_muzero_policy
 (highest count)   (from top-k same top visit count sample 1  from it with gumbel)
 all top-k candidate has same visit count (which is the maximun child visit cound in (a|S))
 so the one selected deterministic must have the highest visit count
 but it might not have the highest q value among candidate (but with higher pr)
 because it is sampled from all the candidates
"""
"""
 it can also be used to seq halving when simulating @action_selection.gumbel_muzero_root_action_selection
 ex. simu_budget = 32, max_consider = 16
 16 [ 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 2 2 2 2 3 3 3 3 ]
 first visit visit count == 0 nodes
 second                     1
 ... etc
"""
def score_considered(considered_visit, #max(root.child_visit_counts)::[B] 
                     gumbel, logits, normalized_qvalues,
                     visit_counts  #root.child_visit_counts::[B,A] 
  ):
  """Returns a score usable for an argmax."""
  # We allow to visit a child, if it is the only considered child.
  low_logit = -1e9
  logits = logits - jnp.max(logits, keepdims=True, axis=-1)    
  #   log exp vi - max (log exp v) = log(exp(vi)/max exp(vi))   normalize it to 0~1
  penalty = jnp.where(visit_counts == considered_visit,
      0, -jnp.inf)   #we want it to only consider multiple nodes with the same largest visit count 
  # id_print(visit_counts)   #(bs,82)  ex. [[0,0,1,0,0,4,2,0,0,0,4,1,0,0,...,0]  ,  ...   [0,0,16,0,0,...16]]
  # id_print(considered_visit)   # (bs,) [4,...,16]
  # id_print(penalty)      #(bs,82)  ex.[[-inf,-,-,-,-,0,-,-,-,-,0,-,-,-,...,-]  ,  ...[-inf,-, 0,-,-,...,0]]
  chex.assert_equal_shape([gumbel, logits, normalized_qvalues, penalty])
  return jnp.maximum(low_logit, gumbel + logits + normalized_qvalues) + penalty


def get_sequence_of_considered_visits(
    max_num_considered_actions,  #0~16+1  The maximum number of considered actions.
    num_simulations  #32  : The total simulation budget.
  ):  #return A tuple with visit counts. Length `num_simulations`.
  """Returns a sequence of visit counts considered by Sequential Halving.
  Sequential Halving is a "pure exploration" algorithm for bandits, introduced
  in "Almost Optimal Exploration in Multi-Armed Bandits":http://proceedings.mlr.press/v28/karnin13.pdf

  The visit counts allows to implement Sequential Halving by selecting the best
  action from the actions with the currently considered visit count.

  ex. simu_num = 30   consider = 8                
  log2(8)=3   ->  8, 4 ,2    ->  30 / 3 = 10
  
num_sim/log2max    10 10 10
num_considered     8  4  2
                   |
                   |
                   |  |
                   |  |  |
num_extra_visits   1  2  5
               10/8 10/4 10/2
  """
  if max_num_considered_actions <= 1:
    return tuple(range(num_simulations))
  log2max = int(math.ceil(math.log2(max_num_considered_actions)))  # ex. 31-> log_2(31)=4.95 -> 5
  sequence = []
  visits = [0] * max_num_considered_actions  
  num_considered = max_num_considered_actions
  while len(sequence) < num_simulations:
    num_extra_visits = max(1, int(num_simulations / (log2max * num_considered)))
    #  ex. num_simu = 30 , num_consider=max_n=8 -> log2max = log_2(8) = 3 -> extra = 30/(3*8) = 1
                                               #round  #1.1      #2.1     #2.2             #3.1 ~ 3.5
                                         #num_extra    10/8=1   10/4=2     2               10/2 = 5  
    for _ in range(num_extra_visits):    #num_consider   8        4        4                  2
      sequence.extend(visits[:num_considered])  # seq [0]*8  [0]8+[1]4  [0]8+[1]4+[2]4  [0]8+[1]4+[2]4+[3]2    
      for i in range(num_considered):         #visits [1]*8  [2]4+[1]4  [3]4+[1]4        [4]2+[3]2+[1]4   
        visits[i] += 1                      
    # Halving the number of considered actions.
    num_considered = max(2, num_considered // 2)    # num_consider = 4
  # id_print(f"max:{sequence[:num_simulations]}")
  return tuple(sequence[:num_simulations])   #  [0]8+[1]4+[2]4+[3]2  +...


def get_table_of_considered_visits(
    max_num_considered_actions,   #16, maximum number of considered actions.
    #it can be smaller than the number of actions. (because of invalid action)
    num_simulations  #ex. 32  The total simulation budget.
  ):  #Returns: A tuple of sequences of visit counts. Shape [max_num_considered_actions + 1, num_simulations].
  """Returns a table of sequences of visit counts."""
  return tuple(
      get_sequence_of_considered_visits(m, num_simulations)
      for m in range(max_num_considered_actions + 1))  #0~16
#                                                                  32/log2(16)=8 
#                                                               8/16=0, 8/8=1, 8/4=2
#  ((0:32),(0:32),)                 [0]8+[1]4+[2]4+[3]2+...       [0]16+[1]8+[2]4+[3]4
#  m=  0     1       ...               8                 ...         16
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