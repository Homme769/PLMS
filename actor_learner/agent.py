import sonnet as snt
import tensorflow as tf

import functools

from tools import (
    AgentOutput,
    PcOutput,
    my_dm_lab,
    conv_variable,
    deconv2d,
    fc_variable,
)
    # conv_variable,
    # deconv2d,
    # fc_variable,

nest = tf.contrib.framework.nest

class Agent(snt.RNNCore):
    """
    Agent with ResNet.
    Actor的代理
    """

    def __init__(self, num_actions, gradient_net=True):
        """
        初始化一个Agent，包含了动作数
        :param num_actions:
        """
        super(Agent, self).__init__(name="agent")

        self._num_actions = num_actions

        with self._enter_variable_scope():
            self._core = tf.nn.rnn_cell.LSTMCell(256)

    def initial_state(self, batch_size):
        """初始化core"""
        return self._core.zero_state(batch_size, tf.float32)

    def pixel_change(self, core_output, reuse=False):
        """
        用来实现pixelchange的head部分
        """
        with tf.variable_scope("pc_deconv", reuse=reuse) as scope:

            W_pc_fc1, b_pc_fc1 = fc_variable([256, 9 * 9 * 32], "pc_fc1")

            W_pc_deconv_v, b_pc_deconv_v = conv_variable(
                [4, 4, 1, 32], "pc_deconv_v", deconv=True
            )
            W_pc_deconv_a, b_pc_deconv_a = conv_variable(
                [4, 4, self._num_actions, 32], "pc_deconv_a", deconv=True
            )

            h_pc_fc1 = tf.nn.relu(tf.matmul(core_output, W_pc_fc1) + b_pc_fc1)
            h_pc_fc1_reshaped = tf.reshape(h_pc_fc1, [-1, 9, 9, 32])

            h_pc_deconv_v = tf.nn.relu(
                deconv2d(h_pc_fc1_reshaped, W_pc_deconv_v, 9, 9, 2) + b_pc_deconv_v
            )
            h_pc_deconv_a = tf.nn.relu(
                deconv2d(h_pc_fc1_reshaped, W_pc_deconv_a, 9, 9, 2) + b_pc_deconv_a
            )
            # Advantage mean
            h_pc_deconv_a_mean = tf.reduce_mean(
                h_pc_deconv_a, reduction_indices=3, keep_dims=True
            )

            # Pixel change Q (output)
            pc_q = h_pc_deconv_v + h_pc_deconv_a - h_pc_deconv_a_mean
            # (-1, 20, 20, action_size)

            # Max Q
            pc_q_max = tf.reduce_max(pc_q, reduction_indices=3, keep_dims=False)
            # (-1, 20, 20)

        return pc_q, pc_q_max

    def _torso(self, input_):
        """
        构建网络的主干部分,获得图像的网络输出和文本的网络输出，以及reward和action， concat之后返回

        input -> frame -> Conv(3, 16) -> 2*res -> Conv(3, 32) -> 2*res -> Conv(3, 32) -> 2*res -> Linear(256)
          |
          | ---> instruction -> _instruction -> out
          |
          | ---> reward -> clip -> out
          |
          | ---> action -> one-hot -> out

        :param input_: 输入
        :return:
        """
        last_action, env_output = input_
        # initial_reward, initial_info, initial_done, initial_observation(frame, string), pixel_control, task_id
        reward, _, _, (frame, instruction), _, _ = env_output

        # Convert to floats. 转化输入的图像
        frame = tf.to_float(frame)

        frame /= 255

        with tf.variable_scope("convnet"):
            conv_out = frame
            # 3*3卷积输出16，接3*3步幅2的最大池化，接2层残差块，每层都是3*3的卷积
            # 之后接3*3卷积输出32，重复上述两次
            # 增加深度
            # @todo
            for i, (num_ch, num_blocks) in enumerate([(16, 2), (32, 2), (32, 2)]):
                # Downscale.
                conv_out = snt.Conv2D(num_ch, 3, stride=1, padding="SAME")(conv_out)
                conv_out = tf.nn.pool(
                    conv_out,
                    window_shape=[3, 3],
                    pooling_type="MAX",
                    padding="SAME",
                    strides=[2, 2],
                )

                # Residual block(s).
                for j in range(num_blocks):
                    with tf.variable_scope("residual_%d_%d" % (i, j)):
                        block_input = conv_out
                        conv_out = tf.nn.relu(conv_out)
                        conv_out = snt.Conv2D(num_ch, 3, stride=1, padding="SAME")(
                            conv_out
                        )
                        conv_out = tf.nn.relu(conv_out)
                        conv_out = snt.Conv2D(num_ch, 3, stride=1, padding="SAME")(
                            conv_out
                        )
                        conv_out += block_input

        conv_out = tf.nn.relu(conv_out)
        conv_out = snt.BatchFlatten()(conv_out)

        conv_out = snt.Linear(256)(conv_out)
        conv_out = tf.nn.relu(conv_out)

        cate_task = snt.Linear(len(my_dm_lab), name="cate_task")(conv_out)

        # Append clipped last reward and one hot last action.
        clipped_reward = tf.expand_dims(tf.clip_by_value(reward, -1, 1), -1)
        one_hot_last_action = tf.one_hot(last_action, self._num_actions)

        # return tf.concat([conv_out, clipped_reward, one_hot_last_action, instruction_out], axis=1)
        return (
            tf.concat([conv_out, clipped_reward, one_hot_last_action], axis=1),
            cate_task,
        )

    def _head(self, core_output, cate_task):
        """
        输出部分的网络结构,输出选择的动作，动作的概率分布，以及最后给的一个baseline

          |--> Linear(num_actions) -> policy_logits -----------------------------| 
          |                                  |                                     |
          |                                  |--------->  sample 1  ---> action ---|
          |                                                                        |
          |---------------->   Linear(1) ------------->   baseline ----------------|
  input---|                                                                        |---- out
          |                                   |------->  q ------------------------|
          |------------->  pix_change  -------|                                    |
          |                                   |-------> max_q ---------------------|                                         
          |                                                                        |
          |------------------------ cate_task -------------------------------------|

        :param core_output: LSTM的Output
        :return:
        """
        # 全连接到num_acitons
        policy_logits = snt.Linear(self._num_actions, name="policy_logits")(core_output)
        # 全连接到1并移除最后一个维度(1,)
        baseline = tf.squeeze(snt.Linear(1, name="baseline")(core_output), axis=-1)

        # Sample an action from the policy.
        new_action = tf.multinomial(policy_logits, num_samples=1, output_dtype=tf.int32)
        new_action = tf.squeeze(new_action, 1, name="new_action")

        q, max_q = self.pixel_change(core_output)

        return AgentOutput(
            new_action, policy_logits, baseline, PcOutput(q, max_q), cate_task
        )

    def _build(self, input_, core_state):
        """
        每一个继承snt.Module的都要实现的函数, __call__调用这里，调用unroll获得返回值

        input -> expand batch -> unroll -> squeeze -> output

        :param input_:
        :param core_state:
        :return:
        """
        action, env_output = input_
        # 将每一个的第1维扩展变为(1, action), (1, env_outputs)
        actions, env_outputs = nest.map_structure(
            lambda t: tf.expand_dims(t, 0), (action, env_output)
        )
        outputs, core_state = self.unroll(actions, env_outputs, core_state)
        # 将返回的output重新缩放回去
        return nest.map_structure(lambda t: tf.squeeze(t, 0), outputs), core_state

    @snt.reuse_variables
    def unroll(self, actions, env_outputs, core_state):
        """
        给定input和core state， 返回output和下一个的core state， 流程

        actions, env_outputs -> _torso -> LSTM(_core) -> _head ---> output
                                           |  |                       |
        core_state ------------------------|  |--------> core_state --'

        :param actions:
        :param env_outputs:
        :param core_state:
        :return:
        """
        _, _, done, _, _, _ = env_outputs

        # 神经网络输出
        torso_outputs, cate_task = snt.BatchApply(self._torso)((actions, env_outputs))
        # print("torso_outputs", torso_outputs)

        initial_core_state = self._core.zero_state(tf.shape(actions)[1], tf.float32)
        core_output_list = []
        # torso的结果输入LSTM网络中继续训练
        for input_, d in zip(tf.unstack(torso_outputs), tf.unstack(done)):
            # 如果结束了采用新的core state
            core_state = nest.map_structure(
                functools.partial(tf.where, d), initial_core_state, core_state
            )
            core_output, core_state = self._core(input_, core_state)
            core_output_list.append(core_output)

        return (
            snt.BatchApply(self._head)(tf.stack(core_output_list), cate_task),
            core_state,
        )

