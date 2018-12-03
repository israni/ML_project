import tensorflow as tf
import tensorflow.contrib.slim as slim
import random
import numpy as np
class networkArchitecture():
	def __init__(self):

		self.ipFrames = tf.placeholder(shape=[None,105,80,4], dtype=tf.float32)
		self.processedFrames = tf.keras.layers.Lambda(lambda x: x/255.0)(self.ipFrames)
		
		self.conv1 = slim.conv2d(inputs=self.processedFrames, num_outputs = 32, kernel_size = [8,8], stride = [4,4], padding = 'VALID', biases_initializer=None) #ReLU is the activation by default in slim.conv2d
		self.conv2 = slim.conv2d(inputs=self.conv1, num_outputs = 64, kernel_size = [4,4], stride = [2,2], padding = 'VALID', biases_initializer=None)
		self.conv3 = slim.conv2d(inputs=self.conv2, num_outputs = 64, kernel_size = [3,3], stride = [1,1], padding = 'VALID', biases_initializer=None)
		self.conv3Flattened = slim.flatten(self.conv3)
		self.fc = slim.fully_connected(inputs=self.conv3Flattened, num_outputs = 512, biases_initializer=None)
		self.opQvalues = slim.fully_connected(inputs=self.fc, num_outputs = 4, biases_initializer = None)
		self.predict = tf.argmax(self.opQvalues,1)

		self.Qtarget = tf.placeholder(shape=[None], dtype = tf.float32)
		self.actions = tf.placeholder(shape=[None], dtype = tf.int32)
		self.actions_onehot = tf.one_hot(self.actions,4, dtype = tf.float32) #second argument env.actions

		self.Qestimate = tf.reduce_sum(tf.multiply(self.opQvalues,self.actions_onehot), axis = 1)

		self.loss = tf.reduce_mean(tf.square(self.Qtarget-self.Qestimate))

		self.trainer = tf.train.AdamOptimizer(learning_rate=0.0001)
		self.trainingStep = self.trainer.minimize(self.loss)


class replayMemoryBuffer():
	def __init__(self, size=50000):
		self.buffer = []
		self.buffer_size = size

	def addSample(self, newExperience):
		totalLength = len(newExperience) + len(self.buffer)
		if(totalLength >= self.buffer_size):
			self.buffer[0:totalLength-self.buffer_size] = []
		self.buffer.extend(newExperience)

	def getSample(self, size):
		print(len(self.buffer),size)
		randomIdx = random.sample(self.buffer,size)
		return np.reshape(np.array(randomIdx,size), [size, 5])


	