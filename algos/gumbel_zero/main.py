import os
from os.path import dirname
import sys
directory = dirname(dirname(dirname(os.path.abspath(__file__))))
sys.path.append(directory)
import haiku as hk;  
import optax
import jax as jx
from jax import jit, vmap, grad
from jax import numpy as jnp
from jax.experimental.host_callback import id_print
import functools
from tqdm import tqdm
#local files
import mctx
from utils.main import get_logger
from jax_environments import ProcMaze
from config import TrainConfig
activation_dict = {"relu": jx.nn.relu, "silu": jx.nn.silu, "elu": jx.nn.elu}
import pyrallis
import time
@pyrallis.wrap()    
def main(config:TrainConfig):
    print(config)
    key = jx.random.PRNGKey(config.seed)
    logger = get_logger(config)
    class P_V_network(hk.Module):
        def __init__(self, config, num_actions, name=None):
            super().__init__(name=name)
            self.n_hidden_units = config.n_hidden_units     #*200
            self.n_layers = config.n_layers   #*3
            self.activation_function = activation_dict[config.activation]  #*elu
            self.num_actions = num_actions
        def __call__(self, x):  #* forward
            #(5,5,4)  (board_h,board_w,channels)  #channels 0~3 player goal wall empty
            x = x.ravel()  #x.shape 100
            for i in range(self.n_layers):   #*3
                x = self.activation_function(hk.Linear(self.n_hidden_units)(x))
            V = hk.Linear(1)(x)[0]  #** last layer    hk.Linear(1)(x) shape = (,)
            # V:  last layer output
            pi_logit = hk.Linear(self.num_actions)(x)
            return V, pi_logit   
    #***  done 
    def get_recurrent_fn(env, V_func):
        #this assumes the agent has access to the exact environment dynamics
        batch_step = vmap(env.step, in_axes=(0,0))    #step (action,env_state)
        batch_V_func = vmap(V_func,in_axes=(None,0))    #v_func (params,obs)
        def recurrent_fn(params, key, actions, env_states):
            V_params = params["V"]
            """
            correspond to 
            mctx.gumbel_muzero_policy(
                params={"V":S["V_target_params"]},
            """
            env_states, obs, rewards, terminals, _ = batch_step(actions, env_states)
            V,pi_logit = batch_V_func(V_params, obs.astype(float))
            recurrent_fn_output = mctx.RecurrentFnOutput(
                reward=rewards,
                discount=(1.0-terminals)*config.discount,    #0.99
                prior_logits=pi_logit,
                value=V
            )
            return recurrent_fn_output, env_states
        return recurrent_fn

    def get_init_fn(env):
        batch_reset = vmap(env.reset)
        def init_fn(key):
            #* environment
            dummy_state = env.reset(key)    #* just for NN init shape checking
            obs = env.get_observation(dummy_state)
            dummy_obs = obs.astype(float)

            key, subkey = jx.random.split(key);     
            subkeys = jx.random.split(subkey, num=config.batch_size);
            env_states = batch_reset(subkeys)   #env.reset(subkeys)
            num_actions = env.num_actions()  #5
            #* agnet nn
            #*  v_net 
            V_net = hk.without_apply_rng(hk.transform(lambda obs: P_V_network(config,num_actions)(obs.astype(float))))
            key, subkey = jx.random.split(key)
            init_V_params = V_net.init(subkey, dummy_obs)  #V_params = V_function's params (W1,b1,W2,b2...)  #their shape are ckeched  by dummy_obs
            V_func = V_net.apply                #    V_func = V_function(config)

            #* v_optim
            opt_v = optax.adamw(learning_rate=config.lr,eps=config.eps_adam, 
                        b1=config.b1_adam, b2=config.b2_adam,weight_decay= config.wd_adam)
            V_opt_state = opt_v.init(init_V_params)  #adam's state   m0  v0
            return env_states, V_func, init_V_params,\
                            opt_v,  V_opt_state,
        return init_fn

    #*   forward   ->  pi_los, v_loss = pi_func(obs,pi_param), V_func(obs,v_param)
    def get_AC_loss(V_func):     
        def AC_loss(V_params, pi_target, V_target, obs):
            V ,pi_logits = V_func(V_params, obs.astype(float))

            pi_loss = jnp.sum(pi_target*(jnp.log(pi_target)-jx.nn.log_softmax(pi_logits)))   #entropyloss =  y*log(y/y_hat)
            V_loss = (V_target-V)**2                         #  MSE

            return jnp.sum(pi_loss+V_loss)
        return AC_loss

    def get_agent_environment_interaction_loop_function(
            env, V_func,  recurrent_fn, opt_v:optax.GradientTransformation, 
            num_actions,  iterations
        ):
        #*                                                    AC_Loss()    V_params, pi_target,V_target, obs
        batch_loss = lambda *x: jnp.mean(vmap(get_AC_loss(V_func), in_axes=(None,0,0,0))(*x))
        loss_grad = grad(batch_loss, argnums=(0))     #* arg V_params
        batch_step = vmap(env.step, in_axes=(0,0))   #batch_step = env.step(action,env_state)
        batch_obs = vmap(env.get_observation)        #batch_obs=get_observation(env_state)
        batch_reset = vmap(env.reset)                #batch_reset=reset(key)
        batch_V_func = vmap(V_func,in_axes=(None,0))   # V_func(V_params , obs)

        def agent_environment_interaction_loop_function(S):
            def loop_function(S, data):
                obs = batch_obs(S["env_states"])
                # print(obs.shape)
                V, pi_logits = batch_V_func(S["V_params"], obs.astype(float))  #V(V_params,obs)

                root = mctx.RootFnOutput(
                prior_logits=pi_logits,
                value=V,
                embedding=S["env_states"]
                )

                S["key"], subkey = jx.random.split(S["key"])
                policy_output = mctx.gumbel_muzero_policy(
                params={"V":S["V_params"]},
                rng_key=subkey,
                root=root,
                recurrent_fn=recurrent_fn,
                num_simulations=config.num_simulations,   #10
                max_num_considered_actions=num_actions,   #5
                qtransform=functools.partial(
                    mctx.qtransform_completed_by_mix_value,
                    use_mixed_value=config.use_mixed_value,  #true
                    value_scale=config.value_scale       #0.1
                ),
                )

                # tree search derived targets for policy and value function
                search_policy = policy_output.action_weights

                search_value = policy_output.search_tree.qvalues(
                    jnp.full(config.batch_size, policy_output.search_tree.ROOT_INDEX))[jnp.arange(config.batch_size), policy_output.action]
                # id_print(search_value)  #shape (1,)   #search_value = qvalues[Root_idx](pi_a)   

                # compute loss gradient compared to tree search targets and update parameters
                #                    AC_Loss()    V_params, pi_target,V_target, obs
                V_grads = loss_grad( S["V_params"], search_policy, search_value, obs)
                v_updates,S["V_opt_state"] = opt_v.update( V_grads, S["V_opt_state"], S["V_params"])#todo can we delet S["V_params??"]
                S["V_params"] = optax.apply_updates(S["V_params"],v_updates)
                S["opt_t"]+=1
                # always take action recommended by tree search
                actions = policy_output.action

                # step the environment
                S["env_states"], obs, reward, terminal, _ = batch_step(actions, S["env_states"])

                # reset environment if terminated
                S["key"], subkey = jx.random.split(S["key"])
                subkeys = jx.random.split(subkey, num=config.batch_size)
                S["env_states"] = jx.tree_util.tree_map(
                    lambda x,y: jnp.where(jnp.reshape(terminal,[terminal.shape[0]]+[1]*(len(x.shape)-1)), x,y), 
                    batch_reset(subkeys),                      # [bs,1,1,1,1,1]   terminal ==True  -> x ,  else y(dont change)
                    S["env_states"])          #{goal, wall_grid, pos, t}                         

                # update statistics for computing average return
                S["episode_return"] += reward
                S["avg_return"] = jnp.where(terminal, 
                # avg_r *0.9 + episode_return*0.1
S["avg_return"]*config.avg_return_smoothing+S["episode_return"]*(1-config.avg_return_smoothing), 
                      S["avg_return"])
                S["episode_return"] = jnp.where(terminal, 0, S["episode_return"])
                S["num_episodes"] = jnp.where(terminal, S["num_episodes"]+1, S["num_episodes"])
                return S, None

            S["key"], subkey = jx.random.split(S["key"])
            S, _ = jx.lax.scan(loop_function, S, None, length=iterations)

            return S
        return agent_environment_interaction_loop_function

    #*********************  main function
    opt_t = 0;   time_step = 0
    avg_return = jnp.zeros(config.batch_size)
    episode_return = jnp.zeros(config.batch_size)
    num_episodes = jnp.zeros(config.batch_size)

    #***  environment
    env = ProcMaze(grid_size=config.env_config.grid_size, 
                   timeout=config.env_config.timeout)  #environment: ProcMaze
    num_actions = env.num_actions()

    key, subkey = jx.random.split(key)

    env_states, V_func, V_params,\
            opt_v, V_opt_state = get_init_fn(env)(subkey)



    recurrent_fn = get_recurrent_fn(env, V_func)

    agent_environment_interaction_loop_function = jit(get_agent_environment_interaction_loop_function(
        env, V_func, recurrent_fn, opt_v, 
        num_actions, config.eval_frequency))

    # run_state contains all information to be maintained and updated in agent_environment_interaction_loop
    run_state_names = ["env_states", "V_opt_state", "V_params" , #"V_target_params",
                    "opt_t", "avg_return", "episode_return", 
                    "num_episodes", "key"]
    var_dict = locals()
    run_state = {name:var_dict[name] for name in run_state_names}

    avg_returns = []
    times = []
    start = time.time()
    real_start = start
    for i in tqdm(range(config.num_steps//config.eval_frequency)):
        # perform a number of iterations of agent environment interaction including learning updates
        run_state = agent_environment_interaction_loop_function(run_state)

        # avg_return is debiased, and only includes batch elements wit at least one completed episode so that it is more meaningful in early episodes
        valid_avg_returns = run_state["avg_return"][run_state["num_episodes"]>0]
        valid_num_episodes = run_state["num_episodes"][run_state["num_episodes"]>0]
        #                                        1-0.9**valid_numepisode
        avg_return = jnp.mean(valid_avg_returns/(1-config.avg_return_smoothing**valid_num_episodes))
        # print("\rRunning Average Return: "+str(avg_return))
        now  = time.time()
        if now - start > 10:  #record every 10 second
            start=start+10
            print(now-real_start)
            logger.log({'avg_return':float(avg_return)})  #tot == config.num_steps==100000  * 128 env
        avg_returns+=[avg_return]

        time_step+=config.eval_frequency
        times+=[time_step]
    logger.finish()
    
if __name__ == "__main__":
    main()