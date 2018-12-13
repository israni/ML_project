import gym
from pprint import pprint
	
env = gym.make('BreakoutDeterministic-v4')
env.reset()
for i in range(1000):
	env.render()
	pprint(vars(env))
	print(env.action_space)
	print(env.unwrapped.get_action_meanings())
	action = input()# TODO, here prompt user for action
	state, reward, done, debug = env.step(int(action))
	
	print(action, reward,done, debug)
	

	if done :
		print("Game over")
		# you can display to the user that he lost the game
		break

env.close()