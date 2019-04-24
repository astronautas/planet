# from atari_wrappers import MaxAndSkipEnv, EpisodicLifeEnv
import numpy as np
import csv
import os

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

    def __init__(self, env, episode_logging_file=None):
        """Make end-of-life == end-of-episode, but only reset on true game over.
        Done by DeepMind for the DQN and co. since it helps value estimation.
        """
        self._env = env

        self.lives = 0
        self.was_real_done  = True

        self.episode_reward = 0.0
        self.timestep = 0
        self.episode = 0

        self.episode_logging_file = episode_logging_file

    def step(self, action):
        obs, reward, done, info = self._env.step(action)
        self.was_real_done = done
        lives = self._env.lives()

        if not(lives is None):
            if lives < self.lives and lives > 0:
                done = True

            self.lives = lives

        if not(self.episode_logging_file is None):
            self.episode_reward += reward
            self.timestep += 1
            self.episode += 1

        return obs, reward, done, info

    def reset(self, **kwargs):
        # print("Called reset")

        """Reset only when lives are exhausted.
        This way all states are still reachable even though lives are episodic,
        and the learner need not know about any of this behind-the-scenes.
        """

        # print(self._env.lives())
        # print(self._env.state().__dir__)
        # print(self.game)

        if not(self._env.lives() is None):
            if self.was_real_done:
                print("Ending episode: ", str(self.timestep) + " " + str(self.episode) + " " + str(self.episode_reward))
                if not(self.episode_logging_file is None):
                    if os.path.isfile(self.episode_logging_file):
                        with open(self.episode_logging_file, 'a+') as f:
                            writer = csv.writer(f)
                            writer.writerow([self.timestep, self.episode, self.episode_reward])
                    else:
                        with open(self.episode_logging_file, 'w') as f:
                            writer = csv.writer(f)
                            writer.writerow([self.timestep, self.episode, self.episode_reward])

                obs = self._env.reset(**kwargs)

                self.episode_reward = 0.0
                self.episode += 1
            else:
                print("Fake reset: ", self._env.lives())
                # no-op step to advance from terminal/lost life state
                obs, _, _, _ = self._env.step(VIZDOOM_NOOP_ACTION)

            self.lives = self._env.lives()
            return obs
        else:
            return self._env.reset(**kwargs)

class ExtractGameState(object):
    def __init__(self, env):
        self._env = env

    def __getattr__(self, name):
        return getattr(self._env, name)

    def state(self):
        return self.game.get_state()

    def lives(self):
        return self.game.get_state().game_variables[0] if not(self.game.get_state() is None) else None

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

def wrap_vizdoom(env, episode_logging_file, episode_life=True, clip_rewards=True, max_and_skip=True):
    env = ExtractGameState(env)

    if episode_life:
        env = EpisodicLifeEnv(env, episode_logging_file=episode_logging_file)

    if clip_rewards:
        env = ClipRewardEnv(env)

    if max_and_skip:
        env = MaxAndSkipEnv(env)

    return env