[maze is cool](https://github.com/kenjyoung/mctx_learning_demo)
# Basic Learning Demo with MCTX
A very basic implementation of AlphaZero style learning with the MCTX framework for Monte Carlo Tree Search in JAX. The included basic_tree_search.py script assumes the agent have access to the exact environment dynamics and does not include any model learning, but rather learns a policy and value function from tree search results as in AlphaZero.

## Usage
Using this repo requires JAX, Haiku, NumPy and wandb. You can start an experiment with 
```bash
#train
python3 basic_tree_search.py -c config.json -o basic_tree_search -s 0
#plot
python3 plot_return.py -o ckpt/basic_tree_search.out
```
The flags have the following meaning:

-c specifies the configuration file, an example of which is provided in config.json.<br>
-o prefix for file names containing output data.<br> 
-s specifies the random seed.<br>

## Preliminary Results
The following plots display running average return as a function of training steps on a small 5x5 procedurally generated Maze environment (implemented as ProcMaze in jax_environments.py). Each time the environment is reset it builds a maze using randomized depth first search. The configuration used for this result is specified in config.json. It should be straightforward to apply to other environments written in jax with a similar interface, though given it uses MCTS with no chance nodes it is not really appropriate for environments with stochastic transitions (randomness in the initial state as in ProcMaze is ok).
<img align="center" src="img/learning_curve.png" width=800>


json
{"num_hidden_layers" : 3,
 "num_hidden_units": 200,
 "V_alpha":0.0001,
 "pi_alpha":0.00004,
 "b1_adam":0.9,
 "b2_adam":0.99,
 "eps_adam":1e-5,
 "wd_adam":1e-6,
 "discount":0.99,
 "use_mixed_value":true,
 "value_scale":0.1,
 "value_target":"maxq",
 "target_update_frequency": 100,
 "batch_size":128,
 "avg_return_smoothing":0.9,
 "num_simulations":50,
 "num_steps":100000,
 "eval_frequency":100,
 "environment": "ProcMaze",
 "env_config":
 {"grid_size":5,
  "timeout":64
 },
 "activation": "elu" || silu || relu
 }

simu50 training time 1:20:00
simu25 training time 0:30:00