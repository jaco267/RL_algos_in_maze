from jax_environments import Asterix, ProcMaze
import jax.numpy as jnp
import jax as jx
import numpy as np
env = ProcMaze()
key = jx.random.PRNGKey(3)
num_actions = env.num_actions()

last_is_gold = None
last_is_enemy = None
last_entity_x = None
last_movement_dir = None

returns=[]
for i in range(100):
	G=0
	key, subkey = jx.random.split(key)
	env_state = env.reset(subkey)
	terminal = False
	while(not terminal):
		a = input('action 0:pass,1:up,2:left,3:down,4:right\n')
		key, subkey = jx.random.split(key)
		if a == 'w':
			action = 1
		elif a == 's':
			action = 3
		elif a == 'a':
			action = 2
		elif a == 'd':
			action = 4
		else:
			action = 0

		
		key, subkey = jx.random.split(key)
		env_state, obs, reward, terminal, _ = env.step( action, env_state)
		goal, wall_grid, pos, t = env_state
		# print('pos',pos)
		
		a = np.array(jnp.where(wall_grid,1,0))
		a[pos[0],pos[1]] = 8
		print(pos,"pos")
		print(a)
		G+=reward
	if(pos[0]>0 and pos[0]<env.grid_size-1):
		assert(last_is_enemy[pos[1]])
		assert(last_entity_x[pos[1]])
	returns+=[G]

print(jnp.mean(jnp.asarray(returns)))
