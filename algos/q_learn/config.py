from dataclasses import dataclass


@dataclass
class EnvConfig:
    grid_size:int = 5
    timeout:int = 64

@dataclass
class TrainConfig:
    project_name:str = "jax_maze"  #wandb name
    algo_name:str="dqn_demo"  #wandb runs name
    seed:int=1
    enable_wandb:bool=True
    ## training 
    batch_size:int=256
    discount_factor:float=0.99  #0.95 will be a little bit more robust, but we want to keep the same gamma with muzero
    train_eps:int=2000
    warm_up_steps:int = 300
    ## network
    n_layers:int=3
    n_hidden_units:int=100   
    activation:str = "elu"  #relu, elu silu
    ## hyper parameters
    lr:float=0.0001  #0.03
    #  adam
    b1_adam:float=0.9
    b2_adam:float=0.99
    eps_adam:float=1e-5
    wd_adam:float=1e-6
    ## q-learn
    epsilon_hlife:int=1500 #epsilon greedy
    buffer_size:int = 1000_000
    ## plot
    avg_return_smoothing:float=0.9
    ## env
    env_config:EnvConfig=EnvConfig