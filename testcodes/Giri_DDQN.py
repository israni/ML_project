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
		self.input = tf.placeholder(shape = [None, 21168], dtype=tf.float32)
		self.inputImage = tf.reshape(self.input, shape = [-1,84,84,3])
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


class experience_buffer():
	def __init__(self, buffer_size=50000):
		self.buffer = []
		self.buffer_size = buffer_size

	def add(self, experience):
		if len(self.buffer) + len(experience) >= self.buffer_size:
			self.buffer[0:len(self.buffer) + len(experience)-self.buffer_size] = []
		self.buffer.extend(experience)

	def sample(self,size):
		return np.reshape(np.array(random.sample(self.buffer, size)), [size, 5])


def processState(states):
	return np.reshape(states, [21168])

def updateTargetGraph(tfVars, tau):
	total_vars = len(tfVars)
	op_holder = []
	for idx,var in enumerate(tfVars[0:total_vars//2]):
		op_holder.append(tfVars[idx+total_vars//2].assign((var.value()*tau) + ((1-tau)*tfVars[idx+total_vars//2].value())))
	return op_holder

def updateTarget(op_holder,sess):
	for op in op_holder:
		sess.run(op)



batch_size = 32 #How many experiences to use for each training step
update_freq = 4 #How often to perform a training step
discountFactor = 0.99
startEpsilon = 1
endEpsilon = 0.1
annealing_steps = 10000. #How many steps of training to reduce startEpsilon to endEpsilon
num_episodes = 500
pre_train_steps = 10000 #How many steps of random actions before training begins
maxEpisodeLength = 50
loadModel = False
path = "./dqn" #Path to save our model
h_size = 512 #The size of the final convolutional layer before splitting it into Advantage and Value streams.
tau = 0.001 #Rate to update target network toward primary network

tf.reset_default_graph()
mainQN = Qnetwork(h_size)
targetQN = Qnetwork(h_size)

init = tf.global_variables_initializer()

saver = tf.train.Saver()
trainables = tf.trainable_variables()

targetOps = updateTargetGraph(trainables,tau)

eBuffer = experience_buffer()

#Set the rate of random action decrease
epsilon = startEpsilon
stepDrop = (startEpsilon - endEpsilon)/annealing_steps

#Lists to contain total rewards and steps per episode
jList = []
rList = []
total_steps = 0

#make a path for our model to saved in

if not os.path.exists(path):
	os.makedirs(path)
		
with tf.Session() as sess:
	sess.run(init)
	if(loadModel):
		print('Loading Model...')
		ckpt = tf.train.get_checkpoint_state(path)
		saver.restore(sess, ckpt.model_checkpoint_path)
	for i in range(num_episodes):
		print("Episode: ", i, end='\r')
		episodeBuffer = experience_buffer()

		state = env.reset() 
		state = processState(state)
		isItTimeToReset = False
		totalRewards = 0
		j = 0
		while j < maxEpisodeLength:
			j+=1
			
			if np.random.rand(1) < epsilon or total_steps < pre_train_steps:
				action = np.random.randint(0,4)
			else:
				action = sess.run(mainQN.predict, feed_dict = {mainQN.input:[state]})[0]
			
			newState, reward, isItTimeToReset = env.step(action)
			newState = processState(newState)
			total_steps += 1
			episodeBuffer.add(np.reshape(np.array([state,action, reward, newState, isItTimeToReset]),[1,5]))

			if(total_steps > pre_train_steps):
				if(epsilon > endEpsilon):
					epsilon -= stepDrop

				if(total_steps%update_freq==0):
					trainBatch = eBuffer.sample(batch_size)

					estimatedQ = sess.run(mainQN.predict, feed_dict = {mainQN.input:np.vstack(trainBatch[:,3])})
					targetQ = sess.run(targetQN.Qout, feed_dict = {targetQN.input:np.vstack(trainBatch[:,3])})
					end_multiplier = -(trainBatch[:,4]-1)

					doubleQ = targetQ[range(batch_size),estimatedQ]
					targetQ = trainBatch[:,2] + (discountFactor*doubleQ*end_multiplier)

					#Update the network with our target values
					_ = sess.run(mainQN.updateModel,\
						feed_dict={mainQN.input:np.vstack(trainBatch[:,0]),  mainQN.targetQ:targetQ, mainQN.actions:trainBatch[:,1]})

					updateTarget(targetOps, sess) #Update the target network toward the primary network.

			totalRewards += reward
			state = newState
			if(isItTimeToReset):
				break

		eBuffer.add(episodeBuffer.buffer)
		jList.append(j)
		rList.append(totalRewards)
		#periodically save the model
		if(i%1000==0):
			saver.save(sess, path + '/model-'+str(i)+'.ckpt')
			print("saved model")

		if(len(rList)%10==0):
			print(total_steps,np.mean(rList[-10:]),epsilon)

	saver.save(sess,path+'/model-'+str(i)+'.ckpt')
print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

rMat = np.resize(np.array(rList),[len(rList)//100,100])
rMean = np.average(rMat,1)
plt.plot(rMean)
plt.show()