from keras.models import Sequential
from keras.layers import Dense, Activation, Flatten, Convolution2D, Permute
import gym
import argparse


# Use argparse to pass in arguments to the python file
p = argparse.ArgumentParser()
p.add_argument('--env_name', type=str, default='Breakout-v0')
p.add_argument('--debug', type = bool, default= 'True')
args = p.parse_args()


# Get the environment from gym
env = gym.make(args.env_name)
num_actions = env.action_space.n

env.reset()
for _ in range(1000):
    env.render()
    env.step(env.action_space.sample()) # take a random action


INPUT_SHAPE = (84, 84)
WINDOW_LENGTH = 4

input_shape = INPUT_SHAPE + (WINDOW_LENGTH,)

if(args.debug):
	print ('debug mode')
	print ('env-name:', args.env_name)
	print('num_actions:',num_actions)
	print('input_shape', input_shape)

model = Sequential()
model.add(Permute((1, 2, 3), input_shape=input_shape))
model.add(Convolution2D(32, (8, 8), strides=(4, 4)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (4, 4), strides=(2, 2)))
model.add(Activation('relu'))
model.add(Convolution2D(64, (3, 3), strides=(1, 1)))
model.add(Activation('relu'))
model.add(Flatten())
model.add(Dense(512))
model.add(Activation('relu'))
model.add(Dense(num_actions))
model.add(Activation('linear'))
print(model.summary())
