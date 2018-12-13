import gym
from helperfns import *
from architecture import *
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib as plt

path = "./models"

def plotData():
	csvFile = open(path+"/trainingData.csv", "r")

	csvFile.close()

if __name__=='__main__':
	env = gym.make('BreakoutDeterministic-v4')
	
	tf.reset_default_graph()
	mainQN = networkArchitecture()
	targetQN = networkArchitecture()

	initOp = tf.global_variables_initializer()
	
	episodeNumber = 10000
	
	filePath = path+'/model'+str(episodeNumber)+'.ckpt'

	saver =tf.train.Saver()
	isTimeToReset = False

	plotData()
	with tf.Session() as sess:
		sess.run(initOp)
		print('Loading Model...')
		saver.restore(sess,filePath)
		state = env.reset()
		state = phi(state) #Process states
		env.render()
		while not isTimeToReset:
			action = sess.run(mainQN.predict, feed_dict = {mainQN.ipFrames:[state]})[0]
			state, reward, isTimeToReset = performAction(env,action)
			state = phi(state)
			print(action,reward,isTimeToReset)
			env.render()
		

	env.close()
	