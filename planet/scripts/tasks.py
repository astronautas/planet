# Copyright 2019 The PlaNet Authors. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import functools
import os
import numpy as np
from planet import control
from planet import networks
from planet import tools
import skimage.transform
import tensorflow as tf
from scipy.interpolate import interp1d
from planet.tools import nested
import cv2
from gym.spaces import MultiDiscrete, Box
import datetime
import math
from gym_vizdoom import (LIST_OF_ENVS, EXPLORATION_GOAL_FRAME, GOAL_REACHING_REWARD)
import vizdoomgym

# from gym_vizdoom.logging.navigation_video_writer import NavigationVideoWriter


Task = collections.namedtuple(
    'Task', 'name, env_ctor, max_length, state_components')


def cartpole_balance(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'balance')
  return Task('cartpole_balance', env_ctor, max_length, state_components)


def cartpole_swingup(config, params):
  action_repeat = params.get('action_repeat', 8)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cartpole', 'swingup')
  return Task('cartpole_swingup', env_ctor, max_length, state_components)


def finger_spin(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity', 'touch']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'finger', 'spin')
  return Task('finger_spin', env_ctor, max_length, state_components)


def cheetah_run(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'cheetah', 'run')
  return Task('cheetah_run', env_ctor, max_length, state_components)


def cup_catch(config, params):
  action_repeat = params.get('action_repeat', 6)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'position', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'ball_in_cup', 'catch')
  return Task('cup_catch', env_ctor, max_length, state_components)


def walker_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'height', 'orientations', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'walker', 'walk')
  return Task('walker_walk', env_ctor, max_length, state_components)

def humanoid_walk(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = [
      'reward', 'com_velocity', 'extremities', 'head_height', 'joint_angles',
      'torso_vertical', 'velocity']
  env_ctor = functools.partial(
      _dm_control_env, action_repeat, max_length, 'humanoid', 'walk')
  return Task('humanoid_walk', env_ctor, max_length, state_components)

def gym_cheetah(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward', 'state']
  env_ctor = functools.partial(
      _gym_env, action_repeat, config.batch_shape[1], max_length,
      'HalfCheetah-v3')
  return Task('gym_cheetah', env_ctor, max_length, state_components)


def gym_racecar(config, params):
  action_repeat = params.get('action_repeat', 2)
  max_length = 1000 // action_repeat
  state_components = ['reward']
  env_ctor = functools.partial(
      _gym_env, action_repeat, config.batch_shape[1], max_length,
      'CarRacing-v0', obs_is_image=True)
  return Task('gym_racing', env_ctor, max_length, state_components)

def gym_seaquest(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 6000 // action_repeat
  state_components = ['reward']
  env_ctor = functools.partial(
      _gym_atari, action_repeat, config.batch_shape[1], max_length,
      'SeaquestNoFrameskip-v4', obs_is_image=True)
  return Task('gym_seaquest', env_ctor, max_length, state_components)

def gym_breakout(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 1600 // action_repeat
  state_components = ['reward']
  env_ctor = functools.partial(
      _gym_atari, action_repeat, config.batch_shape[1], max_length,
      'BreakoutNoFrameskip-v4', obs_is_image=True)
  return Task('gym_breakout', env_ctor, max_length, state_components)

def gym_pong(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 150
  state_components = ['reward']
  env_ctor = functools.partial(
      _gym_atari, action_repeat, config.batch_shape[1], max_length,
      'PongNoFrameskip-v4', obs_is_image=True)
  return Task('gym_pong', env_ctor, max_length, state_components)

def gym_vizdoom_takecover(config, params):
  action_repeat = params.get('action_repeat', 4)
  max_length = 80
  min_length = config.batch_shape[1]
  state_components = ['reward']
  env_ctor = functools.partial(_gym_vizdoom, action_repeat, min_length, max_length, 'VizdoomTakeCover-v0', True)

  return Task('gym_vizdoom_takecover', env_ctor, max_length, state_components)

def _dm_control_env(action_repeat, max_length, domain, task):
  from dm_control import suite

  env = control.wrappers.DeepMindWrapper(suite.load(domain, task), (64, 64))
  env = control.wrappers.ActionRepeat(env, action_repeat)
  env = control.wrappers.MaximumDuration(env, max_length)
  env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
  env = control.wrappers.ConvertTo32Bit(env)

  return env

def _gym_vizdoom(action_repeat, min_length, max_length, name, obs_is_image=False):
  import gym

  env = gym.make(name)

  timestamp = datetime.datetime.now().strftime("%I_%M_%b_%d_%Y")
  env = control.wrap_vizdoom(env, episode_logging_file="/home/lukas/workspace/planet_runs/vizdoom/episodes_info_" + timestamp + ".csv")

  env = control.wrappers.DiscreteToBoxWrapper(env)
  env = control.wrappers.MinimumDuration(env, min_length)
  env = control.wrappers.MaximumDuration(env, max_length) # acting as barrier

  if obs_is_image:
    env = control.wrappers.ObservationDict(env, 'image')
    env = control.wrappers.ObservationToRender(env)
  else:
    env = control.wrappers.ObservationDict(env, 'state')

  env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
  env = control.wrappers.ConvertTo32Bit(env)
  
  return env

def _gym_atari(action_repeat, min_length, max_length, name, obs_is_image=False):
  import gym

  env = gym.make(name)

  env = control.wrap_deepmind(env, episode_life=True, clip_rewards=True, frame_stack=False, scale=False, max_and_skip=True, log_episode_return=True)

  env = control.wrappers.DiscreteToBoxWrapper(env)
  env = control.wrappers.MinimumDuration(env, min_length)
  env = control.wrappers.MaximumDuration(env, max_length) # acting as barrier

  if obs_is_image:
    env = control.wrappers.ObservationDict(env, 'image')
    env = control.wrappers.ObservationToRender(env)
  else:
    env = control.wrappers.ObservationDict(env, 'state')

  env = control.wrappers.PixelObservationsAsGrayscale(env, (64, 64), np.uint8, 'image')
  env = control.wrappers.ConvertTo32Bit(env)

  return env

# def _gym_env(action_repeat, min_length, max_length, name, obs_is_image=False):
#   import gym
#   env = gym.make(name)
#   env = control.wrappers.ActionRepeat(env, action_repeat)
#   env = control.wrappers.NormalizeActions(env)
#   env = control.wrappers.MinimumDuration(env, min_length)
#   env = control.wrappers.MaximumDuration(env, max_length)
#   if obs_is_image:
#     env = control.wrappers.ObservationDict(env, 'image')
#     env = control.wrappers.ObservationToRender(env)
#   else:
#     env = control.wrappers.ObservationDict(env, 'state')
#   env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
#   env = control.wrappers.ConvertTo32Bit(env)
#   return env

# def _gym_env_discrete(action_repeat, min_length, max_length, name, obs_is_image=False):
#   import gym
#   env = gym.make(name)
#   env = control.wrappers.ActionRepeat(env, action_repeat)
#   env = control.wrappers.NormalizeActions(env)
#   env = control.wrappers.MinimumDuration(env, min_length)
#   env = control.wrappers.MaximumDuration(env, max_length)

#   if obs_is_image:
#     env = control.wrappers.ObservationDict(env, 'image')
#     env = control.wrappers.ObservationToRender(env)
#   else:
#     env = control.wrappers.ObservationDict(env, 'state')
#   env = control.wrappers.PixelObservations(env, (64, 64), np.uint8, 'image')
#   env = control.wrappers.ConvertTo32Bit(env)
#   return env
