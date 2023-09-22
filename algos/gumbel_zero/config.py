from dataclasses import dataclass


@dataclass
class EnvConfig:
    grid_size:int = 5
    timeout:int = 64

@dataclass
class TrainConfig:
    algo:str="gumbel_zero_1M_time_net100"
    seed:int=0
    enable_wandb:bool = True
    project_name:str = "jax_maze"  #wandb name
    num_hidden_layers:int=3
    num_hidden_units:int= 100   #200
    V_alpha:float=0.0001
    pi_alpha:float=0.00004
    b1_adam:float=0.9
    b2_adam:float=0.99
    eps_adam:float=1e-5
    wd_adam:float=1e-6
    discount:float=0.99
    use_mixed_value:bool=True
    value_scale:float=0.1
    value_target:str="maxq"
    target_update_frequency:int= 1
    batch_size:int=128
    avg_return_smoothing:float=0.9
    num_simulations:int=5   #10
    num_steps:int=1000_000
    eval_frequency:int=100
    environment:str= "ProcMaze"
    env_config:EnvConfig = EnvConfig
    activation:str= "elu"