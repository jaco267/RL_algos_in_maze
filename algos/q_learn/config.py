from dataclasses import dataclass


@dataclass
class EnvConfig:
    grid_size:int = 5
    timeout:int = 64

@dataclass
class TrainConfig:
    algo:str="dqn_time"
    project_name:str = "jax_maze"  #wandb name
    seed:int=1
    enable_wandb:bool=False
    batch_size:int=256
    
    n_layers:int=3
    n_hidden_units:int=100   #32
    discount_factor:float=0.95
    train_eps:int=1000
    test_eps:int=30
    epsilon_hlife:int=1500
    lr:float=0.03
    dqnft_update_every:int=100
    #how often to copy online parameters to target network in DQN with fixed target
    # environment:str= "ProcMaze"
    warm_up_steps:int = 300
    env_config:EnvConfig=EnvConfig
    avg_return_smoothing:int=0.9## plot
    eval_frequency:int=100