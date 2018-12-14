import gym
from gym import Wrapper, wrappers, spaces
import os
import time

import numpy as np
from collections import deque
from PIL import Image

class NoopResetEnv(gym.Wrapper):
    def __init__(self, env, noop_max=30):
        gym.Wrapper.__init__(self, env)
        self.noop_max = noop_max
        self.override_num_noops = None
        assert env.unwrapped.get_action_meanings()[0] == 'NOOP'

    def _reset(self):
        self.env.reset()
        if self.override_num_noops is not None:
            noops = self.override_num_noops
        else:
            noops = self.unwrapped.np_random.randint(1, self.noop_max + 1)
        assert noops > 0
        obs = None
        for _ in range(noops):
            obs, _, done, _ = self.env.step(0)
            if done:
                obs = self.env.reset()
        return obs

class FireResetEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        assert env.unwrapped.get_action_meanings()[1] == 'FIRE'
        assert len(env.unwrapped.get_action_meanings()) >= 3

    def _reset(self):
        self.env.reset()
        obs, _, done, _ = self.env.step(1)
        if done:
            self.env.reset()
        obs, _, done, _ = self.env.step(2)
        if done:
            self.env.reset()
        return obs

class EpisodicLifeEnv(gym.Wrapper):
    def __init__(self, env):
        gym.Wrapper.__init__(self, env)
        self.lives = 0
        self.was_real_done  = True

    def _step(self, action):
        obs, reward, done, info = self.env.step(action)
        self.was_real_done = done
        lives = self.env.unwrapped.ale.lives()
        if lives < self.lives and lives > 0:
            done = True
        self.lives = lives
        return obs, reward, done, info

    def _reset(self):
        if self.was_real_done:
            obs = self.env.reset()
        else:
            obs, _, _, _ = self.env.step(0)
        self.lives = self.env.unwrapped.ale.lives()
        return obs

class MaxAndSkipEnv(gym.Wrapper):
    def __init__(self, env, skip=4):
        gym.Wrapper.__init__(self, env)
        self._obs_buffer = deque(maxlen=2)
        self._skip       = skip

    def _step(self, action):
        total_reward = 0.0
        done = None
        for _ in range(self._skip):
            obs, reward, done, info = self.env.step(action)
            self._obs_buffer.append(obs)
            total_reward += reward
            if done:
                break
        max_frame = np.max(np.stack(self._obs_buffer), axis=0)

        return max_frame, total_reward, done, info

    def _reset(self):
        self._obs_buffer.clear()
        obs = self.env.reset()
        self._obs_buffer.append(obs)
        return obs

class ClipRewardEnv(gym.RewardWrapper):
    def _reward(self, reward):
        return np.sign(reward)

class WarpFrame(gym.ObservationWrapper):
    def __init__(self, env):
        gym.ObservationWrapper.__init__(self, env)
        self.res = 84
        self.observation_space = spaces.Box(low=0, high=255, shape=(self.res, self.res, 1))

    def _observation(self, obs):
        frame = np.dot(obs.astype('float32'), np.array([0.299, 0.587, 0.114], 'float32'))
        frame = np.array(Image.fromarray(frame).resize((self.res, self.res),
            resample=Image.BILINEAR), dtype=np.uint8)
        return frame.reshape((self.res, self.res, 1))

class FrameStack(gym.Wrapper):
    def __init__(self, env, k):
        gym.Wrapper.__init__(self, env)
        self.k = k
        self.frames = deque([], maxlen=k)
        shp = env.observation_space.shape
        assert shp[2] == 1  
        self.observation_space = spaces.Box(low=0, high=255, shape=(shp[0], shp[1], k))

    def _reset(self):
        ob = self.env.reset()
        for _ in range(self.k): self.frames.append(ob)
        return self._observation()

    def _step(self, action):
        ob, reward, done, info = self.env.step(action)
        self.frames.append(ob)
        return self._observation(), reward, done, info

    def _observation(self):
        assert len(self.frames) == self.k
        return np.concatenate(self.frames, axis=2)

def wrap_deepmind(env, episode_life=True, clip_rewards=True):
    assert 'NoFrameskip' in env.spec.id 
    if episode_life:
        env = EpisodicLifeEnv(env)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = WarpFrame(env)
    if clip_rewards:
        env = ClipRewardEnv(env)
    return env



class Monitor(Wrapper):
    def __init__(self, env, rank=0):
        Wrapper.__init__(self, env=env)
        self.rank = rank
        self.rewards = []
        self.current_metadata = {}  
        self.info = {'reward': 0, 'episode_length': 0}

    def reset(self):
        self.info['reward'] = -1
        self.info['episode_length'] = -1
        self.rewards = []
        return self.env.reset()

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        self.rewards.append(reward)
        if done:
            self.info['reward'] = sum(self.rewards)
            self.info['episode_length'] = len(self.rewards)
        return observation, reward, done, self.info

    
    def monitor(self, is_monitor, is_train, record_video_every=10):
        if is_monitor:
            if is_train:
                self.env = wrappers.Monitor(self.env, 'output', resume=True,
                                                video_callable=lambda count: count % record_video_every == 0)
            else:
                self.env = wrappers.Monitor(self.env, 'test', resume=True,
                                                video_callable=lambda count: count % record_video_every == 0)
        else:
            self.env = wrappers.Monitor(self.env, 'output', resume=True,
                                            video_callable=False)
        self.env.reset()




class GymEnv():
    def __init__(self, env_name, id, seed):
        self.env_name = env_name
        self.rank = id
        self.env = None
        self.seed = seed
        self.make()
        self.gym_env = self.env.env.env.env.env.env.env
        self.monitor = self.env.env.env.env.env.env.monitor

    def make(self):
        env = Monitor(gym.make(self.env_name), self.rank)
        env.seed(self.seed + self.rank)
        self.env = wrap_deepmind(env)
        return env

    def step(self, data):
        observation, reward, done, info = self.env.step(data)
        return observation, reward, done, info

    def reset(self):
        return self.env.reset()

    def get_action_space(self):
        return self.env.action_space

    def get_observation_space(self):
        return self.env.observation_space

    def render(self):
        self.gym_env.render()
