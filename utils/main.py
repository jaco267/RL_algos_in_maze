import wandb
from datetime import datetime
def wandb_init(project_name,name,algos_name,config):
    run = wandb.init(
        project=project_name, # Set the project where this run will be logged
        # Track hyperparameters and run metadata
        name = name,
        group = algos_name,
        config=config
    )    

class void_logger:
    def log(dict):pass
    def finish(): pass


def timestamp():
    return datetime.now().strftime("%B %d, %H:%M:%S")
def get_logger(config):
    if config.enable_wandb:
        wandb.login()
        wandb_init(config.project_name,timestamp(),algos_name=config.algo,config=config)
        logger = wandb
    else:
        logger = void_logger
    return logger