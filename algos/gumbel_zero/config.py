from dataclasses import dataclass


@dataclass
class EnvConfig:
    grid_size:int = 5
    timeout:int = 64

@dataclass
class TrainConfig:
    project_name:str = "jax_maze"  #wandb name
    algo_name:str="gumbel_zero_demo"  #wandb runs name
    seed:int=0
    enable_wandb:bool = True
    ## training 
    batch_size:int=256  #128
    discount:float=0.99
    num_simulations:int=3   #5
    num_steps:int=800_000
    ## network
    n_layers:int=3
    n_hidden_units:int=100   #200
    activation:str= "elu"
    ## hyperparameters
    lr:float=0.0001
    #  adam
    b1_adam:float=0.9
    b2_adam:float=0.99
    eps_adam:float=1e-5
    wd_adam:float=1e-6
    ## muzero
    use_mixed_value:bool=True
    value_scale:float=0.1
    value_target:str="maxq"
    # target_update_frequency:int= 1
    ## plot
    avg_return_smoothing:float=0.9
    eval_frequency:int=100
    ## env
    env_config:EnvConfig = EnvConfig
    