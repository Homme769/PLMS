# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Environments and environment helper classes."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf

nest = tf.contrib.framework.nest

from tools import StepOutput, StepOutputInfo, ALL_LEVELS, PyProcess
from .raw_env import PyProcessDmLab

def create_environment(FLAGS, level_name, seed, task_id, is_test=False):
    """Creates an environment wrapped in a `FlowEnvironment`."""
    if level_name in ALL_LEVELS:
        level_name = 'contributed/dmlab30/' + level_name

    # Note, you may want to use a level cache to speed of compilation of environment maps.
    # See the documentation for the Python interface of DeepMind Lab.
    config = {
        'width': FLAGS.width,
        'height': FLAGS.height,
        'datasetPath': FLAGS.dataset_path,
        'logLevel': 'WARN',
    }

    if is_test:
        config['allowHoldOutLevels'] = 'true'
        # Mixer seed for evalution, see
        # https://github.com/deepmind/lab/blob/master/docs/users/python_api.md
        config['mixerSeed'] = 0x600D5EED

    # 创建一个py_process，里面的proxy传入了PyProcessDmLab以及其构造函数
    p = PyProcess(PyProcessDmLab, level_name, config, FLAGS.num_action_repeats, seed)
    # 使用FlowEnvironment包装
    return FlowEnvironment(p.proxy, task_id)

class FlowEnvironment(object):

    def __init__(self, env, task_id):
        self._env = env
        self._task_id = task_id

    def initial(self):
        with tf.name_scope('flow_environment_initial'):
            initial_reward = tf.constant(0.)
            initial_info = StepOutputInfo(tf.constant(0.), tf.constant(0))  # (episode_return, episode_step)
            initial_done = tf.constant(True)
            initial_observation = self._env.initial()
            initial_pixel_change = tf.zeros([20, 20], dtype=tf.float32)
            initial_task_id = tf.constant(self._task_id)

            initial_output = StepOutput(
                initial_reward,
                initial_info,
                initial_done,
                initial_observation,
                initial_pixel_change,
                initial_task_id)

            # 确保initial_observation已经获取到了之后，才进行之后的操作
            with tf.control_dependencies(nest.flatten(initial_output)):
                initial_flow = tf.constant(0, dtype=tf.int64)
            initial_state = (initial_flow, initial_info)
            return initial_output, initial_state

    def step(self, action, state):
        with tf.name_scope('flow_environment_step'):
            # logging.error("In flow_environment_step")
            flow, info = nest.map_structure(tf.convert_to_tensor, state)

            # Make sure the previous step has been executed before running the next step.
            with tf.control_dependencies([flow]):
                reward, done, observation, pixel_change = self._env.step(action)

            with tf.control_dependencies(nest.flatten(observation)):
                new_flow = tf.add(flow, 1)

            # When done, include the reward in the output info but not in the
            # state for the next step.
            new_info = StepOutputInfo(info.episode_return + reward, info.episode_step + 1)
            # state = flow, (episode_return, episode_step)
            # lambda a, b: tf.where(done, a, b), StepOutputInfo(tf.constant(0.), tf.constant(0)), new_info
            # if done, StepOutputInfo(tf.constant(0.), tf.constant(0)) else new_info
            new_state = new_flow, nest.map_structure(lambda a, b: tf.where(done, a, b),
                                                     StepOutputInfo(tf.constant(0.), tf.constant(0)), new_info)

            output = StepOutput(reward, new_info, done, observation, pixel_change, self._task_id)

            # logging.error("flow_environment_step finished with out: {}".format(output))
            return output, new_state
