import gym
from helperfns import *
from architecture import *
import os
import tensorflow as tf
import numpy as np
import matplotlib as plt

if __name__=='__main__':
	env = gym.make('BreakoutDeterministic-v4')
	frame = env.reset()
	env.render()

	isItTimeToReset = False
	while not isItTimeToReset:
		frame, reward, isItTimeToReset, _ = env.step(env.action_space.sample())
		env.render()

	env.close()
	