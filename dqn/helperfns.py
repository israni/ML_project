import numpy as np
import tensorflow as tf
def phi(states): #Preprocessing
	#states = np.array(states)
	if(len(np.shape(states))==3):
		grayImage = np.mean(states, axis = 2).astype(np.uint8)
		resultTemp = grayImage[::2,::2]
		result = np.zeros((105,80,4))
		for i in range(0,4):
			result[:,:,i] = resultTemp
	else:
		result = np.zeros((105,80,4))
		for i in range(0,4):
			grayImage = np.mean(states[i], axis = 2).astype(np.uint8)
			result[:,:,i] = grayImage[::2,::2]

	return result

def processRewards(reward):
	return np.sign(reward)

def checkModelsEqual(sess):
	vars = tf.trainable_variables() 
	n = int(len(vars)/2)
	result = []
	for i in range(n):
		A = sess.run(vars[i])
		B = sess.run(vars[i+n])
		result.append(np.array_equal(A,B))
	if(all(result)):
		print("equal", result)
	else:
		print("not equal", result)


def updateTargetModel(sess):
	vars = tf.trainable_variables() 
	copy_ops = [vars[ix+len(vars)//2].assign(var.value()) for ix, var in enumerate(vars[0:len(vars)//2])] #[vars[6].assign(vars[1].value())]
	sess.run(copy_ops)
	
	
def performAction(env,action):
	noOfFramesPerState = 4
	newStates = []
	rewards = []
	isItTimeToReset = False
	for i in range(0,noOfFramesPerState):
		if(not isItTimeToReset):
			newState, reward, isItTimeToReset, _ = env.step(action)
		newStates.append(newState)
		rewards.append(reward)

	return newStates, max(rewards), isItTimeToReset

