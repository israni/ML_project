import tensorflow as tf
import os
import numpy as np
import random
import tensorflow as tf
from pprint import pprint
import argparse
import json
import sys
import pickle
from multiprocessing import Process, Pipe




def worker(remote, env_fn_wrapper):
    env = env_fn_wrapper.x()
    while True:
        cmd, data = remote.recv()
        if cmd == 'step':
            ob, reward, done, info = env.step(data)
            total_info = info.copy() 
            if done:
                ob = env.reset()
            remote.send((ob, reward, done, total_info))
        elif cmd == 'reset':
            ob = env.reset()
            remote.send(ob)
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'get_spaces':
            remote.send((env.get_action_space(), env.get_observation_space()))
        elif cmd == 'monitor':
            is_monitor, is_train, record_video_every = data 
            env.monitor(is_monitor, is_train, record_video_every)
        elif cmd == 'render':
            env.render()
        else:
            raise NotImplementedError


class CloudpickleWrapper(object):

    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


class SubprocVecEnv():
    def __init__(self, env_fns):
        """
        env_fns: list of environments to run in sub-processes
        """
        nenvs = len(env_fns)
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(nenvs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

        self.remotes[0].send(('get_spaces', None))
        self.action_space, self.observation_space = self.remotes[0].recv()

    def step(self, actions):
        for remote, action in zip(self.remotes, actions):
            remote.send(('step', action))
        results = [remote.recv() for remote in self.remotes]
        obs, rews, dones, infos = zip(*results)
        return np.stack(obs), np.stack(rews), np.stack(dones), infos

    def reset(self):
        for remote in self.remotes:
            remote.send(('reset', None))
        return np.stack([remote.recv() for remote in self.remotes])

    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def monitor(self, is_monitor=True, is_train=True, record_video_every=10):
        for remote in self.remotes:
            remote.send(('monitor', (is_monitor, is_train, record_video_every)))

    def render(self):
        for remote in self.remotes:
            remote.send(('render', None))

    @property
    def num_envs(self):
        return len(self.remotes)

 
def openai_entropy(logits):
    a0 = logits - tf.reduce_max(logits, 1, keep_dims=True)
    ea0 = tf.exp(a0)
    z0 = tf.reduce_sum(ea0, 1, keep_dims=True)
    p0 = ea0 / z0
    return tf.reduce_sum(p0 * (tf.log(z0) - a0), 1)

def variable_with_weight_decay(kernel_shape, initializer, wd):
    
    w = tf.get_variable('weights', kernel_shape, tf.float32, initializer=initializer)

    collection_name = tf.GraphKeys.REGULARIZATION_LOSSES
    if wd and (not tf.get_variable_scope().reuse):
        weight_decay = tf.multiply(tf.nn.l2_loss(w), wd, name='w_loss')
        tf.add_to_collection(collection_name, weight_decay)
    
    return w

def conv2d_p(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
             initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0):
    with tf.variable_scope(name):
        stride = [1, stride[0], stride[1], 1]
        kernel_shape = [kernel_size[0], kernel_size[1], x.shape[-1], num_filters]

        with tf.name_scope('layer_weights'):
            if w == None:
                w = variable_with_weight_decay(kernel_shape, initializer, l2_strength)
            
        with tf.name_scope('layer_biases'):
            if isinstance(bias, float):
                bias = tf.get_variable('biases', [num_filters], initializer=tf.constant_initializer(bias))
            
        with tf.name_scope('layer_conv2d'):
            conv = tf.nn.conv2d(x, w, stride, padding)
            out = tf.nn.bias_add(conv, bias)

    return out

def orthogonal_initializer(scale=1.0):
    def _ortho_init(shape, dtype, partition_info=None):
        shape = tuple(shape)
        if len(shape) == 2:
            flat_shape = shape
        elif len(shape) == 4: 
            flat_shape = (np.prod(shape[:-1]), shape[-1])
        else:
            raise NotImplementedError
        a = np.random.normal(0.0, 1.0, flat_shape)
        u, _, v = np.linalg.svd(a, full_matrices=False)
        q = u if u.shape == flat_shape else v  
        q = q.reshape(shape)
        return (scale * q[:shape[0], :shape[1]]).astype(np.float32)

    return _ortho_init


def dense_p(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
            bias=0.0):
    n_in = x.get_shape()[-1].value
    with tf.variable_scope(name):
        if w == None:
            w = variable_with_weight_decay([n_in, output_dim], initializer, l2_strength)
        
        if isinstance(bias, float):
            bias = tf.get_variable("layer_biases", [output_dim], tf.float32, tf.constant_initializer(bias))
        
        output = tf.nn.bias_add(tf.matmul(x, w), bias)
        return output



def dense(name, x, w=None, output_dim=128, initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0,
          bias=0.0,
          activation=None, batchnorm_enabled=False, dropout_keep_prob=-1,
          is_training=True
          ):
    with tf.variable_scope(name) as scope:
        dense_o_b = dense_p(name=scope, x=x, w=w, output_dim=output_dim, initializer=initializer,
                            l2_strength=l2_strength,
                            bias=bias)

        if batchnorm_enabled:
            dense_o_bn = tf.layers.batch_normalization(dense_o_b, training=is_training)
            if not activation:
                dense_a = dense_o_bn
            else:
                dense_a = activation(dense_o_bn)
        else:
            if not activation:
                dense_a = dense_o_b
            else:
                dense_a = activation(dense_o_b)

        if dropout_keep_prob != -1:
            dense_o_dr = tf.nn.dropout(dense_a, dropout_keep_prob)
        else:
            dense_o_dr = dense_a

        dense_o = dense_o_dr
    return dense_o


def flatten(x):
    all_dims_exc_first = np.prod([v.value for v in x.get_shape()[1:]])
    o = tf.reshape(x, [-1, all_dims_exc_first])
    return o

def conv2d(name, x, w=None, num_filters=16, kernel_size=(3, 3), padding='SAME', stride=(1, 1),
           initializer=tf.contrib.layers.xavier_initializer(), l2_strength=0.0, bias=0.0,
           activation=None, batchnorm_enabled=False, max_pool_enabled=False, dropout_keep_prob=-1,
           is_training=True):
    with tf.variable_scope(name) as scope:
        conv_o_b = conv2d_p(scope, x=x, w=w, num_filters=num_filters, kernel_size=kernel_size, stride=stride,
                            padding=padding,
                            initializer=initializer, l2_strength=l2_strength, bias=bias)

        if batchnorm_enabled:
            conv_o_bn = tf.layers.batch_normalization(conv_o_b, training=is_training)
            if not activation:
                conv_a = conv_o_bn
            else:
                conv_a = activation(conv_o_bn)
        else:
            if not activation:
                conv_a = conv_o_b
            else:
                conv_a = activation(conv_o_b)

        if dropout_keep_prob != -1:
            conv_o_dr = tf.nn.dropout(conv_a, dropout_keep_prob)
        else:
            conv_o_dr = conv_a

        conv_o = conv_o_dr
        if max_pool_enabled:
            conv_o = max_pool_2d(scope, conv_o_dr)

    return conv_o



class PolicyNetwork():    
    def __init__(self, sess, input_shape, num_actions, reuse=False, is_training=True, name='train'):
        self.initial_state = []
        self.sess = sess
        self.input_shape = input_shape
        self.X_input = None
        self.reuse = reuse
        self.value_s = None
        self.action_s = None
        self.initial_state = []
        with tf.name_scope(name + "policy_input"):
            self.X_input = tf.placeholder(tf.uint8, input_shape)
        with tf.variable_scope("policy", reuse=reuse):
            conv1 = conv2d('conv1', tf.cast(self.X_input, tf.float32) / 255., num_filters=32, kernel_size=(8, 8),
                           padding='VALID', stride=(4, 4),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=is_training)

            conv2 = conv2d('conv2', conv1, num_filters=64, kernel_size=(4, 4), padding='VALID', stride=(2, 2),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=is_training)

            conv3 = conv2d('conv3', conv2, num_filters=64, kernel_size=(3, 3), padding='VALID', stride=(1, 1),
                           initializer=orthogonal_initializer(np.sqrt(2)), activation=tf.nn.relu,
                           is_training=is_training)

            conv3_flattened = flatten(conv3)

            fc4 = dense('fc4', conv3_flattened, output_dim=512, initializer=orthogonal_initializer(np.sqrt(2)),
                        activation=tf.nn.relu, is_training=is_training)

            self.policy_logits = dense('policy_logits', fc4, output_dim=num_actions,
                                       initializer=orthogonal_initializer(np.sqrt(1.0)), is_training=is_training)

            self.value_function = dense('value_function', fc4, output_dim=1,
                                        initializer=orthogonal_initializer(np.sqrt(1.0)), is_training=is_training)

            with tf.name_scope('value'):
                self.value_s = self.value_function[:, 0]

            with tf.name_scope('action'):
                noise = tf.random_uniform(tf.shape(self.policy_logits))
                self.action_s = tf.argmax(self.policy_logits - tf.log(-tf.log(noise)), 1)

    def step(self, observation, *_args, **_kwargs):
        action, value = self.sess.run([self.action_s, self.value_s], {self.X_input: observation})
        return action, value, []  

    def value(self, observation, *_args, **_kwargs):
        return self.sess.run(self.value_s, {self.X_input: observation})




class Model:
    def __init__(self, sess,
                 entropy_coef=0.01, value_function_coeff=0.5, max_gradient_norm=0.5,
                 optimizer_params=None, args=None):
        self.train_batch_size = args.num_envs * args.unroll_time_steps
        self.num_steps = args.unroll_time_steps
        self.num_stack = args.num_stack
        self.actions = None
        self.advantage = None
        self.reward = None
        self.keep_prob = None
        self.is_training = None
        self.step_policy = None
        self.train_policy = None
        self.learning_rate_decayed = None
        self.initial_state = None
        self.X_input_step_shape = None
        self.X_input_train_shape = None
        self.policy_gradient_loss = None
        self.value_function_loss = None
        self.optimize = None
        self.entropy = None
        self.loss = None
        self.learning_rate = None
        self.num_actions = None
        self.img_height, self.img_width, self.num_classes = None, None, None

        self.sess = sess
        self.vf_coeff = value_function_coeff
        self.entropy_coeff = entropy_coef
        self.max_grad_norm = max_gradient_norm

        self.initial_learning_rate = optimizer_params['learning_rate']
        self.alpha = optimizer_params['alpha']
        self.epsilon = optimizer_params['epsilon']    
        

    def build(self, observation_space_params, num_actions):
        self.img_height, self.img_width, self.num_classes = observation_space_params
        self.num_actions = num_actions
        with tf.name_scope('input'):
            self.X_input_train_shape = (
                None, self.img_height, self.img_width, self.num_classes * self.num_stack)
            self.X_input_step_shape = (
                None, self.img_height, self.img_width,
                self.num_classes * self.num_stack)

            self.actions = tf.placeholder(tf.int32, [None])  
            self.advantage = tf.placeholder(tf.float32, [None])
            self.reward = tf.placeholder(tf.float32, [None])  
            self.learning_rate = tf.placeholder(tf.float32, []) 
            self.is_training = tf.placeholder(tf.bool)  
        
        self.step_policy = PolicyNetwork(self.sess, self.X_input_step_shape, self.num_actions, reuse=False,
                                       is_training=False)

        self.train_policy = PolicyNetwork(self.sess, self.X_input_train_shape, self.num_actions, reuse=True,
                                        is_training=self.is_training)

        with tf.variable_scope('train_output'):
            negative_log_prob_action = tf.nn.sparse_softmax_cross_entropy_with_logits(
                logits=self.train_policy.policy_logits,
                labels=self.actions)
            self.policy_gradient_loss = tf.reduce_mean(self.advantage * negative_log_prob_action)
            mse = tf.square(tf.squeeze(self.train_policy.value_function) - self.reward) / 2.
            self.value_function_loss = tf.reduce_mean(mse) 
            self.entropy = tf.reduce_mean(openai_entropy(self.train_policy.policy_logits))
            self.loss = self.policy_gradient_loss - self.entropy * self.entropy_coeff + self.value_function_loss * self.vf_coeff

            with tf.variable_scope("policy"):
                params = tf.trainable_variables()
            grads = tf.gradients(self.loss, params)
            if self.max_grad_norm is not None:
                grads, grad_norm = tf.clip_by_global_norm(grads, self.max_grad_norm)

            grads = list(zip(grads, params))
            optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=self.alpha,
                                                  epsilon=self.epsilon)
            self.optimize = optimizer.apply_gradients(grads)