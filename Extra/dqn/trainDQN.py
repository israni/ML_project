import gym
from helperfns import *
from architecture import *
import os
import sys
import tensorflow as tf
import numpy as np
import matplotlib as plt

if __name__=='__main__':
	env = gym.make('BreakoutDeterministic-v4')
	
	batchSize = 32
	updateFreq = 4
	discountFactor = 0.99
	startEpsilon = 1
	endEpsilon = 0.1
	annealingSteps = 1e6
	numEpisodes = 500000
	preTrainSteps = 1600
	maxEpisodeLength = 50
	loadModel = False
	savedModelPath = "./modelSet2"
	episodeNumber = 20000
	savedFilePath = savedModelPath+'/model'+str(episodeNumber)+'.ckpt'
	targetUpdateFreq = 10000 #steps/actions
	path = "./models"
	
	tf.reset_default_graph()
	mainQN = networkArchitecture()
	targetQN = networkArchitecture()
	initOp = tf.global_variables_initializer()

	saver =tf.train.Saver()

	trainables = tf.trainable_variables()
	experienceBuffer= replayMemoryBuffer(size = 18600)

	epsilon = startEpsilon
	stepDrop = (startEpsilon - endEpsilon)/annealingSteps

	stepCountList = []
	totalRewardList = []
	episodesFinished = 0
	totalSteps = 0
	if not os.path.exists(path):
		os.makedirs(path)

	csvFile = open(path+"/trainingData.csv", "w+")


	with tf.Session() as sess:
		sess.run(initOp)
		if(loadModel):
			print('Loading Model...')
			saver.restore(sess,savedFilePath)
		for episodeNumber in range(numEpisodes):
			print("Episode: ", episodeNumber, end='\r')
			episodeBuffer = replayMemoryBuffer(size = maxEpisodeLength)
			state = env.reset()

			state = phi(state) #Process states

			isTimeToReset = False
			totalRewards = 0
			
			iter = 0
			while( iter < maxEpisodeLength):
				iter = iter + 1
				#print("iteration: ", iter)
				if(np.random.rand(1) < epsilon or totalSteps < preTrainSteps):
					action = np.random.randint(0,4)
				else:
					action = sess.run(mainQN.predict, feed_dict = {mainQN.ipFrames:[state]})[0]


				newState, reward, isTimeToReset = performAction(env,action)
				newState = phi(newState)

				totalSteps = totalSteps + 1

				episodeBuffer.addSample([state,action,reward,newState,isTimeToReset],False) #105 x 80 x 4, 1, 1, 105 x 80 x 4, 1 --> (33600+1+1+33600+(1/8))*4 bytes = 67202.125*4 = 268808.5 bytes = 0.2688085 MB
				#1 GB = 1000 MB, 0.2688085MB/sample --> 3720.1204575 samples/GB --> For 5 GB of buffer size 18600.6022875 ~ 18600 samples!
				if(totalSteps > preTrainSteps):
					if(epsilon > endEpsilon):
						epsilon = epsilon - stepDrop
					if(totalSteps%updateFreq==0):
						trainCurrentStateImages,trainActions,trainRewards,trainNewStateImages,trainIsTimeToReset = experienceBuffer.getSample(batchSize)
						targetOpQvalues = sess.run(targetQN.opQvalues, feed_dict = {targetQN.ipFrames:trainNewStateImages})
						newStateActions = sess.run(targetQN.predict, feed_dict = {targetQN.opQvalues:targetOpQvalues})
						targetQ = sess.run(targetQN.Qestimate, feed_dict = {targetQN.opQvalues:targetOpQvalues, targetQN.actions:newStateActions})
						targetQ[trainIsTimeToReset] = 0
						targetQ = trainRewards + discountFactor*targetQ
						_ = sess.run(mainQN.trainingStep, feed_dict = {mainQN.ipFrames:trainCurrentStateImages, mainQN.Qtarget:targetQ, mainQN.actions:trainActions})
					if(totalSteps%targetUpdateFreq==0):
						updateTargetModel(sess)
	
				totalRewards = totalRewards + reward
				state = newState
				if(isTimeToReset):
					episodesFinished = episodesFinished + 1
					break
			experienceBuffer.addSample(episodeBuffer.buffer,True) #episodeBuffer is of size 5
			stepCountList.append(iter)
			totalRewardList.append(totalRewards)		

			if(episodeNumber%5000==0 or episodeNumber==numEpisodes-1):
				saver.save(sess,path+'/model'+str(episodeNumber)+'.ckpt')
				print("model has been saved")
			if(len(totalRewardList)%100==0 or episodeNumber==numEpisodes-1):
				tr = np.array(totalRewardList)
				sc = np.array(stepCountList)
				temp = np.mean(totalRewardList[-10:])
				trainData = str(totalSteps) + "," + str(temp) + "," + str(epsilon) + "," + str(episodeNumber) + "," + str(episodesFinished) + "," + str(tr.mean()) + "," + str(tr.std()) + "," + str(tr.min()) + "," + str(tr.max()) + "," + str(sc.mean()) + "," + str(sc.std()) + "," + str(sc.min()) + "," + str(sc.max()) +  "\n"
				print('total Steps=',totalSteps, 'mean rewards=', temp, 'epsilon=',epsilon,"TrainData: Rewards mean: %.1f±%.1f," % (tr.mean(), tr.std()), "min: %.1f," % tr.min(), "max: %.1f," % tr.max(),"Steps mean: %.1f±%.1f," % (sc.mean(), sc.std()), "min: %.1f," % sc.min(), "max: %.1f" % sc.max() ) 
				csvFile.write(trainData)
				

	env.close()

		
	csvFile.close()

	
