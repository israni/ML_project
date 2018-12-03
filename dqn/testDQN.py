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

	batchSize = 32
	updateFreq = 4
	discountFactor = 0.99
	startEpsilon = 1
	endEpsilon = 0.1
	annealingSteps = 1e6
	numEpisodes = 500
	preTrainSteps = 30
	maxEpisodeLength = 50
	loadModel = False
	path = "./models"

	tf.reset_default_graph()
	mainQN = networkArchitecture()
	targetQN = networkArchitecture()

	initOp = tf.global_variables_initializer()

	saver =tf.train.Saver()

	trainables = tf.trainable_variables()
	eBuffer = replayMemoryBuffer()

	epsilon = startEpsilon
	stepDrop = (startEpsilon - endEpsilon)/annealingSteps

	stepCountList = []
	totalRewardList = []
	totalSteps = 0

	if not os.path.exists(path):
		os.makedirs(path)

	with tf.Session() as sess:
		sess.run(initOp)
		if(loadModel):
			print('Loading Model...')
			ckpt = tf.train.get_checkpoint_state(path)
			saver.restore(sess,ckpt.model_checkpoint_path)
		for episodeNumber in range(numEpisodes):
			print("Episode: ", episodeNumber, end='\r')
			episodeBuffer = replayMemoryBuffer()
			state = env.reset()
			state = phi(state) #Process states
			isItTimeToReset = False
			totalRewards = 0
			iter = 0
			while( iter < maxEpisodeLength):
				iter = iter + 1

				if(np.random.rand(1) < epsilon or totalSteps < preTrainSteps):
					action = np.random.randint(0,4)
				else:
					action = sess.run(mainQN.predict, feed_dict = {mainQN.ipFrames:[state]})[0]

				newState, reward, isItTimeToReset = env.step(action)
				newState = phi(newState)
				totalSteps = totalSteps + 1
				episodeBuffer.add(np.reshape(np.array([state,action,reward,newState,isItTimeToReset]),[1,5]))

				if(totalSteps > preTrainSteps):
					if(epsilon > endEpsilon):
						epsilon = epsilon - stepDrop

					if(totalSteps%updateFreq==0):
						trainBatch = eBuffer.sample(batchSize)

						estimatedQ = sess.run(mainQN.predict, feed_dict = {mainQN.ipFrames:np.vstack(trainBatch[:,3])})
						targetQ = sess.run(targetQN.opQvalues, feed_dict = {mainQN.ipFrames:np.vstack(trainBatch[:,3])})

						_ = sess.run(mainQN.trainingStep, feed_dict = {mainQN.ipFrames:np.vstack(trainBatch[:,0]), mainQN.Qtarget:targetQ, mainQN.actions:trainBatch[:,1]})

				totalRewards = totalRewards + reward
				state = newState
				if(isItTimeToReset):
					break
			eBuffer.add(episodeBuffer.buffer)
			stepCountList.append(iter)
			totalRewardList.append(totalRewards)
			if(episodeNumber%1000==0):
				saver.save(sess,path+'/model'+str(episodeNumber)+'.ckpt')
				print("model has been saved")
			if(len(totalRewardList)%10==0):
				print('total Steps=',totalSteps, 'mean rewards=', np.mean(totalRewardList[-10:]), 'epsilon=',epsilon)

	saver.save(sess,path,'/model'+str(episodeNumber)+'.ckpt')

print("Percent of succesful episodes: " + str(sum(rList)/num_episodes) + "%")

rMat = np.resize(np.array(rList),[len(rList)//100,100])
rMean = np.average(rMat,1)
plt.plot(rMean)
plt.show()