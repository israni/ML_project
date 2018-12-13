from __future__ import division

import gym
import numpy as np
import random
import tensorflow as tf
import tensorflow.contrib.slim as slim
import matplotlib.pyplot as plt
import scipy.misc
import os

from gridworld import gameEnv

env = gameEnv(partial=False,size=5)


class Qnetwork():
	def __init__(self, hSize):
		#The network recieves a frame from the game, flattened into an array
		#then resizes it and processes it through four convolutional layers.
		self.input = tf.placeholder(shape = [None, 100800], dtype=tf.float32) #210x160x3
		self.inputImage = tf.reshape(self.input, shape = [-1,84,84,4])
		self.conv1 = slim.conv2d(inputs=self.inputImage, num_outputs = 32, kernel_size = [8,8], stride = [4,4], padding = 'VALID', biases_initializer=None)
		self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs = 64, kernel_size = [4,4], stride = [2,2], padding = 'VALID', biases_initializer=None)
		self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs = 64, kernel_size = [3,3], stride = [1,1], padding = 'VALID', biases_initializer=None)
		self.conv4 = slim.conv2d(inputs=self.conv3, num_outputs = hSize, kernel_size = [7,7], stride = [1,1], padding = 'VALID', biases_initializer=None)
		
		#We take the output form the final conv layer and split it into separate advantage and value streams
		self.streamAC, self.streamVC = tf.split(self.conv4, 2, 3)
		self.streamA = slim.flatten(self.streamAC)
		self.streamV = slim.flatten(self.streamVC)
		xavier_init = tf.contrib.layers.xavier_initializer()
		self.AW = tf.Variable(xavier_init([hSize//2, env.actions])) #// is floor division
		self.VW = tf.Variable(xavier_init([hSize//2, 1]))
		self.Advantage = tf.matmul(self.streamA, self.AW)
		self.Value = tf.matmul(self.streamV, self.VW)

		#Then combine them to together to get our final Q-values

		self.Qout = self.Value + tf.subtract(self.Advantage, tf.reduce_mean(self.Advantage, axis=1, keep_dims=True))
		self.predict = tf.argmax(self.Qout,1)

		#Below we obtain the loss by taking the SSD between target and prediction Q values

		self.targetQ = tf.placeholder(shape=[None], dtype = tf.float32)
		self.actions = tf.placeholder(shape=[None], dtype = tf.int32)
		self.actions_onehot = tf.one_hot(self.actions, env.actions, dtype=tf.float32)

		self.Q = tf.reduce_sum(tf.multiply(self.Qout, self.actions_onehot),axis = 1)

		self.td_error = tf.square(self.targetQ - self.Q)

		self.loss = tf.reduce_mean(self.td_error)

		self.trainer = tf.train.AdamOptimizer(learning_rate = 0.0001)
		self.updateModel = self.trainer.minimize(self.loss)





# Import the gym module
import gym

# Create a breakout environment
env = gym.make('BreakoutDeterministic-v4')
# Reset it, returns the starting frame
frame = env.reset()
# Render
env.render()

is_done = False
while not is_done:
  # Perform a random action, returns the new frame, reward and whether the game is over
  frame, reward, is_done, _ = env.step(env.action_space.sample())
  # Render
  env.render()