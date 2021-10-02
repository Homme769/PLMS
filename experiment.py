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

"""Importance Weighted Actor-Learner Architectures."""

from __future__ import absolute_import, division, print_function

import collections
import contextlib
import functools
import logging
import os
import sys
import time

import numpy as np

import tensorflow as tf
from six.moves import range

from actor_learner import Actor, Learner, Agent
from env import create_environment
from tools import DEFAULT_ACTION_SET, my_dm_lab, compute_human_normalized_score, PyProcessHook

os.environ["CUDA_VISIBLE_DEVICES"] = "3"

nest = tf.contrib.framework.nest

flags = tf.flags
FLAGS = tf.flags.FLAGS

flags.DEFINE_string("logdir", "../experiment_logs/", "log directory.")
flags.DEFINE_string("expid", "exp_tmp", "experiment id")
flags.DEFINE_enum("mode", "train", ["train", "test"], "Training or test mode.")

# Flags used for testing.
flags.DEFINE_integer("test_num_episodes", 20, "Number of episodes per level.")

# Flags used for distributed training.
flags.DEFINE_integer("task", -1, "Task id. Use -1 for local training.")
flags.DEFINE_enum(
    "job_name",
    "learner",
    ["learner", "actor"],
    "Job name. Ignored when task is set to -1.",
)

# Training.
flags.DEFINE_integer(
    "total_environment_frames", int(1e8), "Total environment frames to train for."
)
flags.DEFINE_integer("num_actors", 4, "Number of actors.") # actor数量
flags.DEFINE_integer("batch_size", 2, "Batch size for training.") # batch数量
flags.DEFINE_integer("unroll_length", 100, "Unroll length in agent steps.") # 时间步
flags.DEFINE_integer("num_action_repeats", 4, "Number of action repeats.") # 动作重复
flags.DEFINE_float("pc_gamma", 0.9, "PC discount")
flags.DEFINE_integer("seed", 1, "Random seed.")

# Loss
flags.DEFINE_float("entropy_cost", 0, "Entropy cost/multiplier.")
flags.DEFINE_float("baseline_cost", 0.5, "Baseline cost/multiplier.")
flags.DEFINE_float("discounting", 0.99, "Discounting factor.")
flags.DEFINE_enum(
    "reward_clipping",
    "abs_one",
    ["abs_one", "soft_asymmetric"],
    "Reward clipping.",
)

# Environment
# flags.DEFINE_string(
#     "dataset_path",
#     "",
#     "Path to dataset needed for psychlab_*, see "
#     "https://github.com/deepmind/lab/tree/master/data/brady_konkle_oliva2008",
# )
flags.DEFINE_string(
    "level_name",
    "explore_goal_locations_small",
    """Level name or \'dmlab30\' for the full DmLab-30 suite """
    """with levels assigned round robin to the actors.""",
)
flags.DEFINE_integer("width", 84, "Width of observation.")
flags.DEFINE_integer("height", 84, "Height of observation.")

# Optimizer
flags.DEFINE_float("learning_rate", 0.001, "Learning rate.")
flags.DEFINE_float("decay", .99, "RMSProp optimizer decay.")
flags.DEFINE_float("momentum", 0.0, "RMSProp momentum.")
flags.DEFINE_float("epsilon", 0.1, "RMSProp epsilon.")


def is_single_machine():
    return FLAGS.task == -1


@contextlib.contextmanager
def pin_global_variables(device):
    """Pins global variables to the specified device."""

    def getter(getter, *args, **kwargs):
        var_collections = kwargs.get("collections", None)
        if var_collections is None:
            var_collections = [tf.GraphKeys.GLOBAL_VARIABLES]
        if tf.GraphKeys.GLOBAL_VARIABLES in var_collections:
            with tf.device(device):
                return getter(*args, **kwargs)
        else:
            return getter(*args, **kwargs)

    with tf.variable_scope("", custom_getter=getter) as vs:
        yield vs


def start_train(action_set, level_names):
    """
    Train
    """
    print("action set: {}, level_name: {}".format(action_set, level_names))
    # 只在本地进行训练，默认的情况下，这时默认自己即是actor也是learner
    if is_single_machine():
        local_job_device = ""
        shared_job_device = ""
        # actor 和 learner 的标记
        is_actor_fn = lambda i: True
        is_learner = True
        # 全局变量的设备
        global_variable_device = "/gpu"
        server = tf.train.Server.create_local_server()
        filters = []
    else:
        print("Multi Machine, task: {}".format(FLAGS.task))
        local_job_device = "/job:%s/task:%d" % (FLAGS.job_name, FLAGS.task)
        shared_job_device = "/job:learner/task:0"
        is_actor_fn = lambda i: FLAGS.job_name == "actor" and i == FLAGS.task
        is_learner = FLAGS.job_name == "learner"

        # Placing the variable on CPU, makes it cheaper to send it to all the
        # actors. Continual copying the variables from the GPU is slow.
        # 变量放在CPU，因为GPU拷贝变量很慢
        global_variable_device = shared_job_device + "/cpu"

        # 构建集群，包含actor和learner两个任务，其中learner仅包含一个task(论文中描述可以拓展)
        # actor包含num_actors个task
        # cluster = tf.train.ClusterSpec({
        #     'actor': ['localhost:%d' % (8001 + i) for i in range(FLAGS.num_actors)],
        #     'learner': ['localhost:8000']
        # })
        cluster = tf.train.ClusterSpec(
            {
                "actor": ["219.223.251.158:%d" % (17001 + i) for i in range(50)]
                + ["219.223.251.157:%d" % (23001 + i) for i in range(50)]
                + ["219.223.251.133:%d" % (19001 + i) for i in range(32)],
                "learner": ["219.223.251.157:11002"],
            }
        )
        server = tf.train.Server(
            cluster, job_name=FLAGS.job_name, task_index=FLAGS.task
        )
        filters = [shared_job_device, local_job_device]

    # Only used to find the actor output structure.
    # 创建agent，env等，为了获取actor的输出结构，没有其他的动作
    with tf.Graph().as_default():
        # create actor
        agent = Agent(len(action_set))
        # create environment
        env = create_environment(FLAGS, level_names[0], seed=1, task_id=len(my_dm_lab))
        # build actor， get return structure ActorOutput
        actor = Actor(FLAGS=FLAGS)
        structure = actor(agent, env, level_names[0], action_set)
        # 返回的是各个结构体部分合成的结果，这里将其摊平，变成tensor的一个list，统计其结构
        flattened_structure = nest.flatten(structure)
        dtypes = [t.dtype for t in flattened_structure]
        shapes = [t.shape.as_list() for t in flattened_structure]

    # 开始真正的训练过程， pin_global_variables???
    with tf.Graph().as_default(), tf.device(
        local_job_device + "/cpu"
    ), pin_global_variables(global_variable_device):
        tf.set_random_seed(FLAGS.seed)  # Makes initialization deterministic.

        # Create Queue and Agent on the learner.
        with tf.device(shared_job_device):
            queue = tf.FIFOQueue(1, dtypes, shapes, shared_name="buffer")
            agent = Agent(len(action_set))

        # Build actors and ops to enqueue their output.
        enqueue_ops = []
        for i in range(FLAGS.num_actors):
            if is_actor_fn(i):
                # 给每一个actor分配一个环境
                level_name = level_names[i % len(level_names)]
                logging.error("Creating actor %d with level %s", i, level_name)
                env = create_environment(
                    FLAGS, level_name, seed=i + 1, task_id=i % len(my_dm_lab)
                )
                # 给每一个actor构建采集循环
                actor = Actor(FLAGS=FLAGS)
                actor_output = actor(agent, env, level_name, action_set)
                with tf.device(shared_job_device):
                    enqueue_ops.append(queue.enqueue(nest.flatten(actor_output)))
        logging.error("Build Actor and Enqueue Ops Finished")
        # If running in a single machine setup, run actors with QueueRunners
        # (separate threads).
        if is_learner and enqueue_ops:
            tf.train.add_queue_runner(tf.train.QueueRunner(queue, enqueue_ops))

        """ Build learner. """
        if is_learner:
            # Create global step, which is the number of environment frames processed.
            # 全局的总处理帧数
            tf.get_variable(
                "num_environment_frames",
                initializer=tf.zeros_initializer(),
                shape=[],
                dtype=tf.int64,
                trainable=False,
                collections=[tf.GraphKeys.GLOBAL_STEP, tf.GraphKeys.GLOBAL_VARIABLES],
            )

            # Create batch (time major) and recreate structure.
            de_queued = queue.dequeue_many(FLAGS.batch_size)
            de_queued = nest.pack_sequence_as(structure, de_queued)

            # 将output的第1维和第2维对调，之前是(batch, num_roll + 1, ...)现在是（num_roll+1, batch, ...）
            def make_time_major(s):
                return nest.map_structure(
                    lambda t: tf.transpose(t, [1, 0] + list(range(t.shape.ndims))[2:]),
                    s,
                )

            de_queued = de_queued._replace(
                env_outputs=make_time_major(de_queued.env_outputs),
                agent_outputs=make_time_major(de_queued.agent_outputs),
            )

            with tf.device("/gpu"):
                # Using StagingArea allows us to prepare the next batch and send it to
                # the GPU while we're performing a training step. This adds up to 1 step
                # policy lag.
                flattened_output = nest.flatten(de_queued)
                area = tf.contrib.staging.StagingArea(
                    [t.dtype for t in flattened_output],
                    [t.shape for t in flattened_output],
                )
                stage_op = area.put(flattened_output)

                data_from_actors = nest.pack_sequence_as(structure, area.get())

                # Unroll agent on sequence, create losses and update ops.
                learner = Learner(FLAGS=FLAGS)
                output = learner(
                    agent,
                    data_from_actors.agent_state,
                    data_from_actors.env_outputs,
                    data_from_actors.agent_outputs,
                )

        logging.error("Build Learner Finished")
        config = tf.ConfigProto(allow_soft_placement=True, device_filters=filters)
        with tf.train.MonitoredTrainingSession(
            server.target,
            is_chief=is_learner,
            checkpoint_dir=FLAGS.logdir,
            save_checkpoint_secs=600,
            save_summaries_secs=30,
            log_step_count_steps=50000,
            config=config,
            hooks=[PyProcessHook()],
        ) as session:

            if is_learner:
                # Logging.
                logging.error("Learner Start at {}".format(time.time()))
                level_returns = {level_name: [] for level_name in level_names}
                summary_writer = tf.summary.FileWriterCache.get(FLAGS.logdir)

                # Prepare data for first run.
                # actor 开始采集第一波数据
                session.run_step_fn(
                    lambda step_context: step_context.session.run(stage_op)
                )
                logging.error("First collect finished, time: {}\n".format(time.time()))
                # Execute learning and track performance.
                num_env_frames_v = 0
                while num_env_frames_v < FLAGS.total_environment_frames:

                    level_names_v, done_v, infos_v, num_env_frames_v, _ = session.run(
                        (data_from_actors.level_name,) + output + (stage_op,)
                    )

                    logging.error(
                        "finished once learn and collect, frame: {} at {}, time: {}".format(
                            num_env_frames_v, os.getpid(), time.time()
                        )
                    )

                    level_names_v = np.repeat([level_names_v], done_v.shape[0], 0)

                    for level_name, episode_return, episode_step in zip(
                        level_names_v[done_v],
                        infos_v.episode_return[done_v],
                        infos_v.episode_step[done_v],
                    ):
                        episode_frames = episode_step * FLAGS.num_action_repeats

                        logging.error(
                            "Level: %s Episode return: %f", level_name, episode_return
                        )

                        level_name = str(level_name, encoding="utf-8")
                        summary = tf.summary.Summary()
                        summary.value.add(
                            tag=str(level_name) + "/episode_return",
                            simple_value=episode_return,
                        )
                        summary.value.add(
                            tag=str(level_name) + "/episode_frames",
                            simple_value=episode_frames,
                        )
                        summary_writer.add_summary(summary, num_env_frames_v)

                        if FLAGS.level_name == "dmlab30":
                            level_returns[level_name].append(episode_return)

                    if (
                        FLAGS.level_name == "dmlab30"
                        and min(map(len, level_returns.values())) >= 1
                    ):
                        no_cap = compute_human_normalized_score(
                            level_returns, per_level_cap=None
                        )
                        cap_100 = compute_human_normalized_score(
                            level_returns, per_level_cap=100
                        )
                        summary = tf.summary.Summary()
                        summary.value.add(
                            tag="dmlab30/training_no_cap", simple_value=no_cap
                        )
                        summary.value.add(
                            tag="dmlab30/training_cap_100", simple_value=cap_100
                        )
                        summary_writer.add_summary(summary, num_env_frames_v)

                        # Clear level scores.
                        level_returns = {level_name: [] for level_name in level_names}

            else:
                # Execute actors (they just need to enqueue their output).
                logging.error("Is Actor")
                while True:
                    session.run(enqueue_ops)


def start_test(action_set, level_names):
    """Test."""

    level_returns = {level_name: [] for level_name in level_names}
    with tf.Graph().as_default():
        agent = Agent(len(action_set))
        outputs = {}
        for i, level_name in enumerate(level_names):
            env = create_environment(
                FLAGS, level_name, seed=i + 1, is_test=True, task_id=7
            )
            actor = Actor(FLAGS=FLAGS)
            outputs[level_name] = actor(agent, env, level_name, action_set)

        with tf.train.SingularMonitoredSession(
            checkpoint_dir=FLAGS.logdir, hooks=[PyProcessHook()]
        ) as session:
            for level_name in level_names:
                logging.error("Testing level: %s", level_name)
                while True:
                    done_v, infos_v = session.run(
                        (
                            outputs[level_name].env_outputs.done,
                            outputs[level_name].env_outputs.info,
                        )
                    )
                    returns = level_returns[level_name]

                    returns.extend(infos_v.episode_return[1:][done_v[1:]])
                    if len(returns):
                        logging.error(
                            "round: {}, return: {}".format(len(returns), returns[-1])
                        )
                    if len(returns) >= FLAGS.test_num_episodes:
                        logging.error("Mean episode return: %f", np.mean(returns))
                        logging.error("returns: %s", returns)
                        break

    if FLAGS.level_name == "dmlab30":
        no_cap = compute_human_normalized_score(level_returns, per_level_cap=None)
        cap_100 = compute_human_normalized_score(level_returns, per_level_cap=100)
        logging.error("No cap.: %f Cap 100: %f", no_cap, cap_100)


def main(_):
    tf.logging.set_verbosity(logging.ERROR)

    action_set = DEFAULT_ACTION_SET
    if FLAGS.level_name == "dmlab30" and FLAGS.mode == "train":
        level_names = list(my_dm_lab.keys())
    elif FLAGS.level_name == "dmlab30" and FLAGS.mode == "test":
        level_names = list(my_dm_lab.values())
    else:
        level_names = [FLAGS.level_name]

    if FLAGS.expid:
        FLAGS.logdir += FLAGS.expid

    if FLAGS.mode == "train":
        start_train(action_set, level_names)
    else:
        start_test(action_set, level_names)


if __name__ == "__main__":
    tf.app.run()
