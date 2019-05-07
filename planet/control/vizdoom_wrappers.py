# from atari_wrappers import MaxAndSkipEnv, EpisodicLifeEnv
import numpy as np
import csv
import os
import imageio
import os
import collections
from vizdoom import GameVariable
import cv2

VIZDOOM_NOOP_ACTION = None

class MaxAndSkipEnv(object):
    def __getattr__(self, name):
        return getattr(self._env, name)

    def __init__(self, env, skip=4):
        """Return only every `skip`-th frame"""

        self._env = env

        # most recent raw observations (for max pooling across time steps)
        self._obs_buffer = np.zeros((2,) + env.observation_space.shape, dtype=np.uint8)
        self._skip = skip

    def step(self, action):
        """Repeat action, sum reward, and max over last observations."""
        total_reward = 0.0
        done = None

        for i in range(self._skip):
            obs, reward, done, info = self._env.step(action)

            if i == self._skip - 2: self._obs_buffer[0] = obs
            if i == self._skip - 1: self._obs_buffer[1] = obs

            total_reward += reward

            if done:
                break

        # Note that the observation on the done=True frame
        # doesn't matter
        max_frame = self._obs_buffer.max(axis=0)

        return max_frame, total_reward, done, info

    def reset(self, **kwargs):
        return self._env.reset(**kwargs)

class EpisodicLifeEnv(object):
    def __getattr__(self, name):
        return getattr(self._env, name)

    def __init__(self, env, episode_logging_file=None, video_logging_file=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        self._env = env

        self.was_real_done  = True

        self.episode_reward = 0.0
        self.episode_reward_exp_avg = 0.0
        self.episode_average_buffer = collections.deque(maxlen=100)
        self.timestep = 0
        self.episode = 0

        self.episode_logging_file = episode_logging_file
        self.video_logging_file = video_logging_file

        self.record_video_every = 20
        self.recording = False
        self.obs_buffer = collections.deque(maxlen=5000)

        self.phase = None

        self.recorded_already = False

    def step(self, action):
        obs, reward, done, info = self._env.step(action)

        if self._phase == "simulate" and not(self.episode_logging_file is None):
            self.episode_reward += reward
            self.timestep += 1

        if self._phase == "simulate" and self._env.state() != None and self.lives > 0:
            buff = np.transpose(self._env.state().screen_buffer, [1, 2, 0])
            self.obs_buffer.append(buff)
        
        if self._phase == "simulate" and (self.lives == 0 and self.recorded_already == False):
            self.recorded_already = True
            self.finalize_episode_recording()
            self.obs_buffer.clear()
        
        if self._phase == "simulate" and self.lives > 0:
            self.recorded_already = False

        return obs, reward, done, info

    def finalize_episode_recording(self):
        self.episode_average_buffer.append(self.episode_reward)

        print("-------")
        print("Ended episode")
        print("Timestep: ", self.timestep)
        print("Episode: ", self.episode)
        print("Reward: ", self.episode_reward)
        print("100 episode avg: ", np.mean(np.array(self.episode_average_buffer)))
        print("100 episode std: ", np.std(np.array(self.episode_average_buffer)))
        print("-------")

        self.record_video()

        if not(self.episode_logging_file is None):
            if os.path.isfile(self.episode_logging_file):
                with open(self.episode_logging_file, 'a+') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.timestep, self.episode, self.episode_reward])
            else:
                with open(self.episode_logging_file, 'w') as f:
                    writer = csv.writer(f)
                    writer.writerow([self.timestep, self.episode, self.episode_reward])

        self.episode_reward = 0.0
        self.episode = 0

    def reset(self, **kwargs):
        self._phase = kwargs.get("phase", None)
        return self._env.reset()
    
    def record_video(self):
        video_path = self.video_logging_file
        
        print("Trying to record")
        if len(self.obs_buffer):
            print("------")
            if os.path.isfile(video_path):
                print("Removing existing mp4")
                os.remove(video_path)

            imageio.mimwrite(video_path, self.obs_buffer, fps=25.0)
            print("Recorded MP4 episode!!!! at", video_path)
            print("------")
        else:
            print("Queue empty")
class ExtractGameState(object):
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def state(self):
        return self.game.get_state()
    
    @property
    def lives(self):
        if not(self.started):
            return None

        health = self.game.get_game_variable(GameVariable.HEALTH)
        return max(health, 0) # as it sometimes gets negative

class ClipRewardEnv(object):
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        return obs, self.reward(reward), done, info

    def reward(self, reward):
        """Bin reward to {+1, 0, -1} by its sign."""
        return np.sign(reward)

def wrap_vizdoom(env, episode_logging_file, video_logging_file, episode_life=True, clip_rewards=True, max_and_skip=True):
    env = ExtractGameState(env)

    if episode_life:
        env = EpisodicLifeEnv(env, episode_logging_file=episode_logging_file, video_logging_file=video_logging_file)

    # if clip_rewards:
    #     env = ClipRewardEnv(env)

    # if max_and_skip:
    #     env = MaxAndSkipEnv(env)

    return env