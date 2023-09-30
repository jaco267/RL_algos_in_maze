from os import path
from os.path import dirname
import sys
file_path = path.abspath(__file__)
directory = dirname(dirname(dirname(file_path)))
sys.path.append(directory)

from dataclasses import dataclass
import pyrallis
from dqn_agent import DQNAgent
import jax as jx
import jax.numpy as jnp
# local file
from utils.main import get_logger
import time

from config import TrainConfig


def run(logger,env,agent,key, training=True, warm_up_steps = 300, ep_steps=20,  **kwargs):
    ep_rewards = [];    #ep_losses = [];     
    total_steps = 0
    avg_return = 0; beta = kwargs['avg_return_smoothing']
    start = time.time()
    for i_episode in range(int(ep_steps+warm_up_steps)):
        env_states = env.reset(key)
        ep_return = 0; ep_loss = [];   done = False
        t = 0
        while not done:
            # Step environment and add to buffer
            observation = env.get_observation(env_states)
            action = agent.act(observation, training)
            env_states, next_observation, reward, done, _ = env.step(action,env_states)
            if training:
                agent.buffer.append((observation, action, reward, next_observation, done))
            observation=next_observation
            if training and i_episode > warm_up_steps:
                loss = agent.update(kwargs["batch_size"])
                ep_loss.append(loss)
            # Update counters:
            ep_return += reward
            t += 1
            total_steps += 1

        ep_rewards.append(ep_return)
        # Log appropriatley
        epsilon = agent.epsilon
        avg_return =  avg_return*beta+ep_return*(1-beta)
        if training and i_episode < warm_up_steps:
            print(f"warmup {i_episode}/{warm_up_steps}",end='\r')
            start = time.time()
        elif training and i_episode>= warm_up_steps:
            now  = time.time()
            if now - start > 10:
                start=start+10
                logger.log({'avg_return':int(avg_return)})
            real_i = i_episode-warm_up_steps
            ep_mean_loss = jnp.array(ep_loss).mean()
            print(f"Training: Episode {real_i}/{ep_steps}, Total Steps {total_steps}, Reward {avg_return}"+\
                  f", Loss {ep_mean_loss:.4f}, Epsilon {epsilon:.4f}",end='\r')
            # ep_losses.append(ep_mean_loss)
        else:
            print(f"Testing: Episode {i_episode}, Total Steps {total_steps}, Reward {avg_return}"+\
                  f", Epsilon {epsilon:.4f}")
    if not training:
        print( f"Test: Avg reward over {i_episode + 1} episodes {jnp.array(ep_rewards).mean():0.3f}")
    return agent

from jax_environments import ProcMaze
@pyrallis.wrap()    
def main(config: TrainConfig):
    logger = get_logger(config)
    # Create environment specified
    env = ProcMaze(grid_size=config.env_config.grid_size,
                   timeout=config.env_config.timeout)
    key = jx.random.PRNGKey(config.seed)
    key, subkey = jx.random.split(key)
    num_actions = env.num_actions()
    agent = DQNAgent(num_actions=num_actions,env=env,key=subkey,config=config, **vars(config))
    # Training
    agent = run(logger,env, agent,key=subkey, ep_steps=config.train_eps, **vars(config))
    # Testing
    # run(logger, env, agent,key=subkey, training=False, ep_steps=config.test_eps, **vars(config) )
    logger.finish()
if __name__ == "__main__":
    main()
