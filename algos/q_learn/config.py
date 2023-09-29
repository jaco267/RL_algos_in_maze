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
    enable_wandb:bool=False

    ## training 
    batch_size:int=256
    discount_factor:float=0.95
    train_eps:int=1000
    warm_up_steps:int = 300
    ## network
    n_layers:int=3
    n_hidden_units:int=100   #32
    ## hyper parameters
    epsilon_hlife:int=1500
    lr:float=0.0001#0.03
    dqnft_update_every:int=100
    ## plot
    avg_return_smoothing:float=0.9
    ## env
    env_config:EnvConfig=EnvConfig