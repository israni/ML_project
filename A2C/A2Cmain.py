import tensorflow as tf
from easydict import EasyDict as edict
import os
import numpy as np
import random
import tensorflow as tf
from pprint import pprint
import argparse
import json
import sys
from architecture import *
from helper_functions import GymEnv
import time
from tqdm import tqdm
import csv 

class LearningRateDecay(object):
	def __init__(self, v, nvalues, lr_decay_method):
		self.n = 0.
		self.v = v
		self.nvalues = nvalues

		def constant(p):
			return 1

		def linear(p):
			return 1 - p

		lr_decay_methods = {
			'linear': linear,
			'constant': constant
		}

		self.decay = lr_decay_methods[lr_decay_method]

	def value(self):
		current_value = self.v * self.decay(self.n / self.nvalues)
		self.n += 1.
		return current_value

	def get_value_for_steps(self, steps):
		return self.v * self.decay(steps / self.nvalues)

class Trainer():

	def save(self):
		print("Saving model...")
		self.saver.save(self.sess, self.args.train_dir, self.global_step_tensor)
		print("Model saved")

	def _load_model(self):
		latest_checkpoint = tf.train.latest_checkpoint(self.args.train_dir)
		if latest_checkpoint:
			print("Loading model checkpoint {} ...\n".format(latest_checkpoint))
			self.saver.restore(self.sess, latest_checkpoint)
			print("Checkpoint loaded\n\n")
		else:
			print("No checkpoints available!\n\n")

	def __init_global_saver(self):
		self.saver = tf.train.Saver(max_to_keep=self.args.max_to_keep)

	def _init_model(self):
		
		self.__init_global_step()
		self.__init_global_time_step()
		self.__init_cur_epoch()
		self.__init_global_saver()
		self.init = tf.group(tf.global_variables_initializer(), tf.local_variables_initializer())
		self.sess.run(self.init)

	def __init_cur_epoch(self):
		with tf.variable_scope('cur_epoch'):
			self.cur_epoch_tensor = tf.Variable(0, trainable=False, name='cur_epoch')
			self.cur_epoch_input = tf.placeholder('int32', None, name='cur_epoch_input')
			self.cur_epoch_assign_op = self.cur_epoch_tensor.assign(self.cur_epoch_input)

	def __init_global_step(self):
		with tf.variable_scope('global_step'):
			self.global_step_tensor = tf.Variable(0, trainable=False, name='global_step')
			self.global_step_input = tf.placeholder('int32', None, name='global_step_input')
			self.global_step_assign_op = self.global_step_tensor.assign(self.global_step_input)

	def __init_global_time_step(self):
		with tf.variable_scope('global_time_step'):
			self.global_time_step_tensor = tf.Variable(0, trainable=False, name='global_time_step')
			self.global_time_step_input = tf.placeholder('int32', None, name='global_time_step_input')
			self.global_time_step_assign_op = self.global_time_step_tensor.assign(self.global_time_step_input)

	def __init__(self, sess, model, r_discount_factor=0.99,
				 lr_decay_method='linear', args=None):
		self.model = model
		self.args = args
		self.sess = sess
		
		self.save_every = 20000
		self.sess = sess
		self.num_steps = self.model.num_steps
		self.cur_iteration = 0
		self.global_time_step = 0
		self.observation_s = None
		self.states = None
		self.dones = None
		self.env = None

		self.num_iterations = int(self.args.num_iterations)

		self.gamma = r_discount_factor

		self.learning_rate_decayed = LearningRateDecay(v=self.args.learning_rate,
													   nvalues=self.num_iterations * self.args.unroll_time_steps * self.args.num_envs,
													   lr_decay_method=lr_decay_method)

	def train(self, env):
		self._init_model()
		self._load_model()

		self.env = env
		self.observation_s = np.zeros(
			(env.num_envs, self.model.img_height, self.model.img_width, self.model.num_classes * self.model.num_stack),
			dtype=np.uint8)
		self.observation_s = self.__observation_update(self.env.reset(), self.observation_s)

		self.states = self.model.step_policy.initial_state
		self.dones = [False for _ in range(self.env.num_envs)]

		tstart = time.time()
		loss_list = np.zeros(100, )
		policy_loss_list = np.zeros(100, )
		value_loss_list = np.zeros(100,)
		values_list = np.zeros(20*100,)
		rewards_list = np.zeros(20*100,)
		policy_entropy_list = np.zeros(100, )
		fps_list = np.zeros(100, )
		list1 = []
		list2 = []
		list3 = []
		arr_idx = 0
		start_iteration = self.global_step_tensor.eval(self.sess)
		self.global_time_step = self.global_time_step_tensor.eval(self.sess)
		csvFile = open("trainData/trainingData.csv", "w+")
		
		for iteration in tqdm(range(start_iteration, self.num_iterations + 1, 1), initial=start_iteration,
							  total=self.num_iterations):

			self.cur_iteration = iteration

			obs, states, rewards, masks, actions, values, cum_rewards, steps_per_episode, mean_rewards_per_episode = self.__rollout()
			loss, policy_loss, value_loss, policy_entropy = self.__rollout_update(obs, states, rewards, masks, actions,
																				  values)
			loss_list[arr_idx] = loss
			policy_loss_list[arr_idx] = policy_loss
			value_loss_list[arr_idx] = value_loss
			policy_entropy_list[arr_idx] = policy_entropy
			list1.append(cum_rewards)
			list2.append(steps_per_episode)
			list3.append(mean_rewards_per_episode)
			values_list[arr_idx*20:arr_idx*20+20] = values
			rewards_list[arr_idx*20:arr_idx*20+20] = rewards
			
			nseconds = time.time() - tstart
			fps_list[arr_idx] = int((iteration * self.num_steps * self.env.num_envs) / nseconds)
			
			
			self.global_step_assign_op.eval(session=self.sess, feed_dict={
				self.global_step_input: self.global_step_tensor.eval(self.sess) + 1})
			arr_idx += 1

			if not arr_idx % 100:
				mean_loss = np.mean(loss_list)
				mean_fps = np.mean(fps_list)
				mean_pe = np.mean(policy_entropy_list)
				mean_pl = np.mean(policy_loss_list)
				mean_vl = np.mean(value_loss_list)
				mean_rewards = np.mean(rewards_list)
				mean_values = np.mean(values_list)
				mean_list1 = np.sum(list1)
				mean_list2 = np.mean(list2)
				mean_list3 = np.mean(list3)
				trainData = str(iteration) + "," + str(mean_loss)[:8] + "," + str(mean_pe)[:8] + "," + str(mean_pl)[:8] + "," + str(mean_vl)[:8] + "," + str(mean_values)[:8] + "," + str(mean_rewards)[:8] + "," + str(mean_list1)[:8] + "," +str(mean_list2)[:8]+ "," +str(mean_list3)[:8] + "\n"
				csvFile.write(trainData)
				print('Iteration:' + str(iteration) + '; loss: ' + str(mean_loss)[:8] + '; entropy: ' + str(mean_pe)[:8] + '; ploss: ' + str(mean_pl)[:8] + '; vloss: ' + str(mean_vl)[:8]+ 
				'; cum_rewards: ' + str(mean_list1)[:8] + '; steps/epi: ' + str(mean_list2)[:8]+ '; rewards/epi: ' + str(mean_list3)[:8] + 
				'; values: ' + str(mean_values)[:8] + '; rewards: ' + str(mean_rewards)[:8] + '; fps: ' + str(mean_fps))
				arr_idx = 0
			if iteration % self.save_every == 0:
				self.save()
		self.env.close()
		csvFile.close()

	def test(self, total_timesteps, env):
		self._init_model()
		self._load_model()

		states = self.model.step_policy.initial_state

		dones = [False for _ in range(env.num_envs)]

		observation_s = np.zeros(
			(env.num_envs, self.model.img_height, self.model.img_width,
			 self.model.num_classes * self.model.num_stack),
			dtype=np.uint8)
		observation_s = self.__observation_update(env.reset(), observation_s)

		for _ in tqdm(range(total_timesteps)):
			actions, values, states = self.model.step_policy.step(observation_s, states, dones)
			observation, rewards, dones, _ = env.step(actions)
			for n, done in enumerate(dones):
				if done:
					observation_s[n] *= 0
			observation_s = self.__observation_update(observation, observation_s)
		env.close()

	def __rollout_update(self, observations, states, rewards, masks, actions, values):
		
		advantages = rewards - values
		for step in range(len(observations)):
			current_learning_rate = self.learning_rate_decayed.value()
		feed_dict = {self.model.train_policy.X_input: observations, self.model.actions: actions,
					 self.model.advantage: advantages,
					 self.model.reward: rewards, self.model.learning_rate: current_learning_rate,
					 self.model.is_training: True}
		if states != []:
			feed_dict[self.model.S] = states
			feed_dict[self.model.M] = masks
		loss, policy_loss, value_loss, policy_entropy, _ = self.sess.run(
			[self.model.loss, self.model.policy_gradient_loss, self.model.value_function_loss, self.model.entropy,
			 self.model.optimize],
			feed_dict
		)
		return loss, policy_loss, value_loss, policy_entropy

	def __observation_update(self, new_observation, old_observation):
		updated_observation = np.roll(old_observation, shift=-1, axis=3)
		updated_observation[:, :, :, -1] = new_observation[:, :, :, 0]
		return updated_observation

	def __discount_with_dones(self, rewards, dones, gamma):
		discounted = []
		r = 0
		for reward, done in zip(rewards[::-1], dones[::-1]):
			r = reward + gamma * r * (1. - done) 
			discounted.append(r)
		return discounted[::-1]


	def __rollout(self):
		train_input_shape = (self.model.train_batch_size, self.model.img_height, self.model.img_width,
							 self.model.num_classes * self.model.num_stack)

		mb_obs, mb_rewards, mb_actions, mb_values, mb_dones = [], [], [], [], []
		mb_states = self.states
		cum_rewards = []
		steps_per_episode = []
		mean_rewards_per_episode = []
		for n in range(self.num_steps):
			actions, values, states = self.model.step_policy.step(self.observation_s, self.states, self.dones)

			mb_obs.append(np.copy(self.observation_s))
			mb_actions.append(actions)
			mb_values.append(values)
			mb_dones.append(self.dones)

			observation, rewards, dones, info = self.env.step(actions)
			decode = lambda x: x if x!=-1 else 0
			for i in range(len(info)):
				cum_rewards.append(decode(info[i]['reward']))
				steps_per_episode.append(decode(info[i]['episode_length']))
				temp = decode(info[i]['reward'])/float(info[i]['episode_length'])
				mean_rewards_per_episode.append(temp)

			self.global_time_step += 1
			self.global_time_step_assign_op.eval(session=self.sess, feed_dict={
				self.global_time_step_input: self.global_time_step})

			self.states = states
			self.dones = dones
			for n, done in enumerate(dones):
				if done:
					self.observation_s[n] *= 0
			self.observation_s = self.__observation_update(observation, self.observation_s)
			mb_rewards.append(rewards)
		mb_dones.append(self.dones)

		mb_obs = np.asarray(mb_obs, dtype=np.uint8).swapaxes(1, 0).reshape(train_input_shape)
		mb_rewards = np.asarray(mb_rewards, dtype=np.float32).swapaxes(1, 0)
		mb_actions = np.asarray(mb_actions, dtype=np.int32).swapaxes(1, 0)
		mb_values = np.asarray(mb_values, dtype=np.float32).swapaxes(1, 0)
		mb_dones = np.asarray(mb_dones, dtype=np.bool).swapaxes(1, 0)
		mb_masks = mb_dones[:, :-1]
		mb_dones = mb_dones[:, 1:]
		last_values = self.model.step_policy.value(self.observation_s, self.states, self.dones).tolist()

		for n, (rewards, dones, value) in enumerate(zip(mb_rewards, mb_dones, last_values)):
			rewards = rewards.tolist()
			dones = dones.tolist()
			if dones[-1] == 0:
				rewards = self.__discount_with_dones(rewards + [value], dones + [0], self.gamma)[:-1]
			else:
				rewards = self.__discount_with_dones(rewards, dones, self.gamma)
			mb_rewards[n] = rewards

		mb_rewards = mb_rewards.flatten()
		mb_actions = mb_actions.flatten()
		mb_values = mb_values.flatten()
		mb_masks = mb_masks.flatten()
		return mb_obs, mb_states, mb_rewards, mb_masks, mb_actions, mb_values, cum_rewards, steps_per_episode, mean_rewards_per_episode

class A2C:
	def __init__(self, sess, args):
		self.args = args
		self.model = Model(sess,
						   optimizer_params={
							   'learning_rate': args.learning_rate, 'alpha': 0.99, 'epsilon': 1e-5}, args=self.args)
		self.trainer = Trainer(sess, self.model, args=self.args)
		self.env_class = A2C.env_name_parser(self.args.env_class)

	def train(self):
		env = A2C.make_all_environments(self.args.num_envs, self.env_class, self.args.env_name,
										self.args.env_seed)

		print("\n\nBuilding the model...")
		self.model.build(env.observation_space.shape, env.action_space.n)
		print("Model is built successfully\n\n")

		with open(self.args.env_name + '.pkl', 'wb') as f:
			pickle.dump((env.observation_space.shape, env.action_space.n), f, pickle.HIGHEST_PROTOCOL)

		print('Training...')
		try:
			try:
				if self.args.record_video_every != -1:
					env.monitor(is_monitor=True, is_train=True, record_video_every=self.args.record_video_every)
			except:
				pass
			self.trainer.train(env)
		except KeyboardInterrupt:
			print('Error occured..\n')
			self.trainer.save()
			env.close()

	def test(self, total_timesteps):
		observation_space_shape, action_space_n = None, None
		try:
			with open(self.args.env_name + '.pkl', 'rb') as f: observation_space_shape, action_space_n = pickle.load(f)
		except:
			print(
				"Environment or checkpoint data not found. Make sure that env_data.pkl is present in the experiment by running training first.\n")
			exit(1)

		env = self.make_all_environments(num_envs=1, env_class=self.env_class, env_name=self.args.env_name,
										 seed=self.args.env_seed)

		self.model.build(observation_space_shape, action_space_n)

		print('Testing...')
		try:
			try:
				if self.args.record_video_every != -1:
					env.monitor(is_monitor=True, is_train=False, record_video_every=self.args.record_video_every)
				else:
					env.monitor(is_monitor=True, is_train=False, record_video_every=20)
			except:
				pass
			self.trainer.test(total_timesteps=total_timesteps, env=env)
		except KeyboardInterrupt:
			print('Error occured..\n')
			env.close()

	@staticmethod
	def __env_maker(env_class, env_name, i, seed):
		def __make_env():
			return env_class(env_name, i, seed)

		return __make_env

	@staticmethod
	def make_all_environments(num_envs=4, env_class=None, env_name="SpaceInvaders", seed=42):
		try:
			tf.set_random_seed(seed)
			np.random.seed(seed)
			random.seed(seed)
		except:
			return ImportError
		return SubprocVecEnv([A2C.__env_maker(env_class, env_name, i, seed) for i in range(num_envs)])

	@staticmethod
	def env_name_parser(env_name):
		envs_to_class = {'GymEnv': GymEnv}

		if env_name in envs_to_class:
			return envs_to_class[env_name]
		raise ValueError("There is no environment with this name. Make sure that the environment exists.")


if __name__ == '__main__':	
	config_args = {'num_envs': 4, 
				   'env_class': 'GymEnv', 
				   'env_name': 'BreakoutNoFrameskip-v4', 
				   'env_seed': 42, 
				   'unroll_time_steps': 5, 
				   'num_stack': 4, 
				   'num_iterations': 100000.0, 
				   'learning_rate': 0.0007, 
				   'reward_discount_factor': 0.99, 
				   'max_to_keep': 4, 
				   'record_video_every': -1, 
				   'to_train': False, 
				   'to_test': True}
	config_args = edict(config_args)

	tf.reset_default_graph()

	config = tf.ConfigProto(allow_soft_placement=True,
							intra_op_parallelism_threads=config_args.num_envs,
							inter_op_parallelism_threads=config_args.num_envs)
	config.gpu_options.allow_growth = True
	sess = tf.Session(config=config)

	config_args.train_dir = 'trainData/'
	config_args.test_dir = 'testData/'
	dirs = [config_args.train_dir, config_args.test_dir]
	try:
		for dir_ in dirs:
			if not os.path.exists(dir_):
				os.makedirs(dir_)
		print("Directories created")
	except Exception as err:
		print("Creating directories error: {0}".format(err))
		exit(-1)

	a2c = A2C(sess, config_args)

	if(len(sys.argv)>1):
		if(sys.argv[1]=='-t'):
			a2c.test(total_timesteps=10000000)
		else:
			a2c.train()
	else:
		a2c.train()
