import imp
import tensorflow as tf

import deepmind_lab
import gym
import os
import cv2
import time
import numpy as np

from tools import calc_pixel_change

class LocalLevelCache(object):
    """Local level cache. 使用level cache对部分环境进行加速"""

    def __init__(self, cache_dir='/tmp/level_cache'):
        self._cache_dir = cache_dir
        tf.gfile.MakeDirs(cache_dir)

    def fetch(self, key, pk3_path):
        """
        从cache中获取key，找到key的话返回true，并将cache复制到pk3_path
        :param key:
        :param pk3_path:
        :return:
        """
        path = os.path.join(self._cache_dir, key)
        if tf.gfile.Exists(path):
            tf.gfile.Copy(path, pk3_path, overwrite=True)
            return True
        return False

    def write(self, key, pk3_path):
        path = os.path.join(self._cache_dir, key)
        if not tf.gfile.Exists(path):
            # Copy the cached file DeepMind Lab has written to the cache directory
            tf.gfile.Copy(pk3_path, path)




"""           PyProcessDmLab
封装了deepMind Lab的类，对外提供initial和step函数，同时提供_tensor_specs来给PyProcess调用，返回对应的TensorSpecs
"""


# is open render mode, save video at video/level_name-time.mp4.')
RENDER = False
RENDER_WIDTH = 320
RENDER_HEIGHT = 240
RENDER_FPS = 30
RENDER_FORMAT = cv2.VideoWriter_fourcc(*'mp4v')
RENDER_PATH = '/root/MyScalable/Video/'

"""           PyProcessDmLab
封装了deepMind Lab的类，对外提供initial和step函数，同时提供_tensor_specs来给PyProcess调用，返回对应的TensorSpecs
"""


class PyProcessDmLab(object):
    """DeepMind Lab wrapper for PyProcess."""

    def __init__(self, level, config, num_action_repeats, seed, runfiles_path=None, level_cache=None):
        """
        初始化一个封装的DmLab对象
        :param level: 环境名称
        :param config: 配置信息
        # config 的定义：
        # width               horizontal resolution of the observation frames              320
        # height              vertical resolution of the observation frames                240
        # fps                 frames per second                                            60
        # levelDirectory      optional path to level directory (relative paths             ''
        #                     are relative to game_scripts/levels)
        # appendCommand       Commands for the internal Quake console*                     ''
        # mixerSeed           value combined with each of the seeds fed to the             '0'
        #                     environment to define unique subsets of seeds
        :param num_action_repeats: 动作重复的次数，每次更新执行给定的动作num_action_repeats帧
        :param seed: 随机数
        :param runfiles_path:
        :param level_cache: 加速环境的cache
        """
        self._num_action_repeats = num_action_repeats
        self._level = level
        self._random_state = np.random.RandomState(seed=seed)
        if runfiles_path:
            deepmind_lab.set_runfiles_path(runfiles_path)
        config = {k: str(v) for k, v in config.items()}
        # 指定observations
        # RGB_INTERLEAVED 返回 RGB [ h, w, 3 ]
        # INSTR 返回 string
        self._observation_spec = ['RGB_INTERLEAVED', 'INSTR']
        self._env = deepmind_lab.Lab(
            level=level,
            observations=self._observation_spec,
            config=config,
            level_cache=level_cache,
        )
        self.last_state = None
        # render env
        if RENDER:
            config_render = config
            config_render['height'] = str(RENDER_HEIGHT)
            config_render['width'] = str(RENDER_WIDTH)
            self._render_env = deepmind_lab.Lab(
                level=level,
                observations=self._observation_spec,
                config=config_render,
                level_cache=level_cache,
            )
            self._make_new_video()
        

    def _make_new_video(self):
            self._video_name = RENDER_PATH + self._level.split('/')[-1] + "-" + str(time.time()) + '.mp4'
            self._video = cv2.VideoWriter(self._video_name, RENDER_FORMAT, RENDER_FPS, (RENDER_WIDTH, RENDER_HEIGHT))
            print('Game for level ', self._level, 'will be record in file: ', self._video_name)

    def _reset(self):
        seed = self._random_state.randint(0, 2 ** 31 - 1)
        self._env.reset(seed=seed)
        if RENDER:
            self._render_env.reset(seed=seed)

    def _observation(self):
        # 返回指定的obs
        d = self._env.observations()
        if RENDER:
            frame = self._render_env.observations()['RGB_INTERLEAVED']
            self._video.write(frame)
        return [d[k] for k in self._observation_spec]

    def initial(self):
        self._reset()
        return self._observation()

    def step(self, action):
        # logging.error("STEP IN PyProcessDmLab with {} at process {}".format(action, os.getpid()))
        reward = self._env.step(action, num_steps=self._num_action_repeats)
        # logging.error("Reward {}".format(reward))
        if RENDER:
            self._render_env.step(action, num_steps=self._num_action_repeats)
        done = np.array(not self._env.is_running())
        if done:
            # observation为动作之后的obs，如果游戏结束，那么为下一局的开始环境
            # logging.error("Done")
            self._reset()
            if RENDER:
                self._video.release()
                self._make_new_video()

        observation = self._observation()
        reward = np.array(reward, dtype=np.float32)
        # 分为20 * 20 个分区 计算和前一个相比的像素变化值
        pixel_change = np.zeros((20, 20), dtype=np.float32)
        if self.last_state is not None:
            pixel_change = calc_pixel_change(self.last_state, observation[0])
        self.last_state = observation[0]
        return reward, done, observation, pixel_change

    def close(self):
        self._env.close()
        if RENDER:
            self._render_env.close()

    @staticmethod
    def _tensor_specs(method_name, unused_kwargs, constructor_kwargs):
        """Returns a nest of `TensorSpec` with the method's output specification."""
        """在调用函数时，返回一个和返回值一样的TensorSpec"""
        width = constructor_kwargs['config'].get('width', 320)
        height = constructor_kwargs['config'].get('height', 240)
        # RGB_INTERLEAVED 和 INSTR
        observation_spec = [
            tf.contrib.framework.TensorSpec([height, width, 3], tf.uint8),
            tf.contrib.framework.TensorSpec([], tf.string),
        ]

        if method_name == 'initial':
            return observation_spec
        elif method_name == 'step':
            # reward, done, obs
            return (
                tf.contrib.framework.TensorSpec([], tf.float32),
                tf.contrib.framework.TensorSpec([], tf.bool),
                observation_spec,
                tf.contrib.framework.TensorSpec([20, 20], tf.float32)
            )



class PyProcessAtari(object):

    def __init__(self, level, config, num_action_repeats, seed):
        self._env = gym.make(level)
        self._num_action_repeats = num_action_repeats

    def _reset(self):
        self._env.reset()

    def initial(self):
        return self._reset()

    def step(self, action):
        reward = 0
        done = False
        obs = None
        for i in range(self._num_action_repeats):
            obs, reward_, done, _ = self._env.step(action)
            reward += reward_
        if done:
            self._reset()
        reward = np.array(reward, dtype=np.float32)
        return reward, done, obs

    def close(self):
        self._env.close()

    @staticmethod
    def _tensor_specs(method_name, kwargs, constructor_kwargs):
        obs_shape = kwargs['shape']
        obs_type = kwargs['dtype']
        observation_spec = [
            tf.contrib.framework.TensorSpec(obs_shape, obs_type),
        ]

        if method_name == 'initial':
            return observation_spec
        elif method_name == 'step':
            # reward, done, obs
            return (
                tf.contrib.framework.TensorSpec([], tf.float32),
                tf.contrib.framework.TensorSpec([], tf.bool),
                observation_spec,
            )
