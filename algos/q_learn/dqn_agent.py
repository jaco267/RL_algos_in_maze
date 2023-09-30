'''
code is borrowed from 
https://github.com/erees1/jax-rl/blob/main/src/agents/dqn_agents.py
'''

from jax import random
import jax as jx
import jax.numpy as jnp
import haiku as hk
from collections import deque
import optax
from jax import random, value_and_grad, jit, vmap
import jax.numpy as jnp
import logging
import optax
logger = logging.getLogger()
activation_dict = {"relu": jx.nn.relu, "silu": jx.nn.silu, "elu": jx.nn.elu}
def batch_func(predict_func):
    #                              params observation
    f = vmap(predict_func, in_axes=(None, 0))
    return f
def mse_loss(func, params, X, Y):
    preds = func(params, X)
    lo = jnp.mean(jnp.square(preds - Y))
    if jnp.isnan(lo):
        raise ValueError(f"Loss went to nan, the predictions were {preds} and the target was {Y}")
    return lo
def optim_update(func, params, X, Y, optim=None,optim_state=None):
    l, grads = value_and_grad(mse_loss, argnums=1)(func, params, X, Y)
    # print(grads.keys())
    if jnp.isnan(grads['dqn_net/linear']['w']).any():
        raise ValueError(f"gradient went to nan, the inputs were {X} and the target was {Y}")
    updates,optim_state = optim.update(grads,optim_state,params)
    params = optax.apply_updates(params,updates)
    return l, params,optim_state#jit(grad_descent, static_argnums=2)(params, grads, step_size)
class DQN_net(hk.Module):
    def __init__(self,config,num_actions):
        super().__init__(name=None)
        self.n_hidden_units = config.n_hidden_units     #*100
        self.n_layers = config.n_layers   #*3
        self.activation_function = activation_dict[config.activation]  #*elu
        self.num_actions = num_actions
    def __call__(self, x):
        x = x.ravel()
        for i in range(self.n_layers):   #*3
            x = self.activation_function(hk.Linear(self.n_hidden_units)(x))
        output = hk.Linear(self.num_actions)(x)
        return output
class ReplayBuffer:
    def __init__(self, maxlen, seed=0):
        self.max_len = maxlen
        self.buf = deque(maxlen=maxlen)
        self.key = random.PRNGKey(seed)
    def sample_batch(self, batch_size):
        self.key = random.split(self.key)[0]
        idxs = random.randint(self.key, (batch_size,), 0, len(self))
        batch = [self[idx] for idx in idxs]
        # Each item to be its own tensor of len batch_size
        b = list(zip(*batch))
        buf_mean = 0
        buf_std = 1
        #     [ obs, actions, r, next_obs, dones, ]
        return [(jnp.asarray(t) - buf_mean) / buf_std for t in b]
    def append(self, x):
        self.buf.append(x)
    def __getitem__(self, idx):
        return self.buf[idx]
    def __len__(self):
        return len(self.buf)
    
class DQNAgent():
    def __init__(self, num_actions=None,env=None,key=None,config=None, lr=0.001, epsilon_hlife=500, epsilon=1, epsilon_min=0.2,
         buffer_size=1000000, discount_factor=0.90, seed=0, **kwargs,):
        super().__init__()
        # Options
        self.lr = lr; self.epsilon_init = epsilon;    self.buffer_size = buffer_size
        self.discount_factor = discount_factor;   self.epsilon_min = epsilon_min
        self.epsilon_decay = 2 ** (-1 / epsilon_hlife)
        # Setup key for initilisation
        self.key = random.PRNGKey(seed)
        self.key, subkey = jx.random.split(self.key)
        self.buffer = ReplayBuffer(self.buffer_size)
        self.steps_trained = 0
        dummy_state = env.reset(subkey)
        dummy_obs = env.get_observation(dummy_state).astype(float)
        V_net = hk.without_apply_rng(hk.transform(lambda obs: DQN_net(config,num_actions)(obs.astype(float))))
        
        self.params = V_net.init(subkey, dummy_obs)
        self.V_func = V_net.apply                #    V_func = V_function(config)
        #layer_spec=[4,32,32,32,2]
        self.predict = lambda observations: self.V_func(self.params, observations)
        self.batched_predict = lambda observations: batch_func(self.V_func)(
            self.params, observations)
        self.num_actions  = num_actions
        self.optim=optax.chain(
            optax.clip(1.0),
            optax.adamw(learning_rate=config.lr,eps=config.eps_adam, 
                        b1=config.b1_adam, b2=config.b2_adam,weight_decay= config.wd_adam)
        )
        self.opt_state = self.optim.init(self.params)
    def act(self, observation, explore=True):
        self.key, subkey = random.split(self.key)
        self.epsilon = (self.epsilon_decay ** self.steps_trained) * (
            self.epsilon_init - self.epsilon_min) + self.epsilon_min
        if explore and random.uniform(self.key) < self.epsilon:
            action = random.randint(subkey, (), 0, self.num_actions)
        else:
            Q = self.V_func(self.params, observation)
            action = jnp.argmax(Q)
        return int(action)

    def update(self, batch_size):
        def get_Q_for_actions(params, observations):
            """Calculate Q values for action that was taken"""
            pred_Q_values = batch_func(self.V_func)(params, observations)
            pred_Q_values = index_Q_at_action(pred_Q_values, actions)
            return pred_Q_values
        ( obs, actions, r, next_obs, dones, ) = self.buffer.sample_batch(batch_size)

        max_next_Q_values = self.get_max_Q_values(next_obs)
        target_Q_values = self.get_target_Q_values(r, dones, max_next_Q_values)

        #  Caclulate loss and perform gradient descent
        loss, self.params,self.opt_state = optim_update(  get_Q_for_actions, self.params, obs, target_Q_values,optim=self.optim,optim_state=self.opt_state  )

        self.steps_trained += 1
        return loss

    def get_max_Q_values(self, next_obs):
        """Calculate max Q values for next state"""
        next_Q_values = self.batched_predict(next_obs)
        max_next_Q_values = jnp.max(next_Q_values, axis=-1)
        return max_next_Q_values

    def get_target_Q_values(self, rewards, dones, max_next_Q_values):
        """Calculate target Q values based on discounted max next_Q_values"""
        target_Q_values = (
            rewards + (1 - dones) * self.discount_factor * max_next_Q_values
        )
        return target_Q_values

def index_Q_at_action(Q_values, actions):
    # Q_values [bsz, n_actions]
    # Actions [bsz,]
    idx = jnp.expand_dims(actions, -1)
    # pred_Q_values [bsz,]
    pred_Q_values = jnp.take_along_axis(Q_values, jnp.array(idx, dtype=jnp.int32), -1).squeeze()
    return pred_Q_values

