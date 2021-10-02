import tensorflow as tf

from tools import DEFAULT_ACTION_SET

"""
损失函数类，计算所有的损失函数，返回结构
"""


class Loss:
    def __init__(
        self,
        use_cate=True,  # 是否使用分类任务
        use_pixel_control=True,  # 是否使用控制辅助任务
        baseline_cost=None,  # v 的输出在loss 中的比重
        entropy_cost=None,  # 熵的比重
        FLAGS=None,
    ):
        self.cate = use_cate
        self.pc = use_pixel_control
        self.baseline_cost = baseline_cost
        self.entropy_cost = entropy_cost
        self.FLAGS = FLAGS

        assert self.baseline_cost != None
        assert self.entropy_cost != None
        assert self.FLAGS != None

    def __call__(
        self,
        pg_advantages=None,  # PG 的优势值
        base_advantage=None,  # baseline 的优势值
        pg_logits=None,  # pg logit
        cate_logits=None,  # 分类的 logit
        actions=None,
        cate_labels=None,
        cate_cost=None,
        pix_cost=None,
        pc_net_q=None,
        pc_net_q_max=None,
        pixel_change=None,
    ):
        self._valid_parameters(
            pg_advantages,
            base_advantage,
            pg_logits,
            cate_logits,
            actions,
            cate_labels,
            pc_net_q,
            cate_cost,
            pix_cost,
            pc_net_q_max,
            pixel_change,
        )
        loss = self._compute_policy_gradient_loss(
            logits=pg_logits, actions=actions, advantages=pg_advantages
        )
        loss += self.baseline_cost * self._compute_baseline_loss(
            advantages=base_advantage
        )
        loss += self.entropy_cost * self._compute_entropy_loss(logits=pg_logits)

        if self.cate:
            loss += cate_cost * self._compute_multi_cate_loss(
                labels=cate_labels, logit=cate_logits
            )

        if self.pc:
            loss += pix_cost * self._compute_pixel_control_loss(
                pc_net_q=pc_net_q,
                pc_net_q_max=pc_net_q_max,
                actions=actions,
                pixel_change=pixel_change,
            )
        return loss

    def _valid_parameters(
        self,
        pg_advantages,
        base_advantage,
        pg_logits,
        cate_logits,
        actions,
        cate_labels,
        pc_net_q,
        cate_cost,
        pix_cost,
        pc_net_q_max,
        pixel_change,
    ):
        """确保每个参数都是正确的，之后计算Loss不会出错"""
        assert pg_advantages != None
        assert base_advantage != None
        assert pg_logits != None
        assert actions != None

        if self.cate:
            assert cate_cost != None
            assert cate_labels != None
            assert cate_logits != None
        if self.pc:
            assert pc_net_q != None
            assert pix_cost != None
            assert pc_net_q_max != None
            assert pixel_change != None

    def _compute_baseline_loss(self, advantages):
        """
        值函数的loss
        """
        # Loss for the baseline, summed over the time dimension.
        # Multiply by 0.5 to match the standard update rule:
        # d(loss) / d(baseline) = advantage
        return 0.5 * tf.reduce_sum(tf.square(advantages))

    def _compute_entropy_loss(self, logits):
        policy = tf.nn.softmax(logits)
        log_policy = tf.nn.log_softmax(logits)
        # -p log p
        entropy_per_timestep = tf.reduce_sum(-policy * log_policy, axis=-1)
        return -tf.reduce_sum(entropy_per_timestep)

    def _compute_policy_gradient_loss(self, logits, actions, advantages):
        """
        PG Loss
        """
        # 计算log pi(a|s)
        cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=actions, logits=logits
        )
        advantages = tf.stop_gradient(advantages)
        # log pi(a|s) * (r + discount * v_t+1 - v_t)
        policy_gradient_loss_per_timestep = cross_entropy * advantages
        # sum
        return tf.reduce_sum(policy_gradient_loss_per_timestep)

    def _compute_multi_cate_loss(self, labels, logit):
        # labels = tf.expand_dims(labels, axis=-1)
        multi_ce_per_timestep = tf.nn.sparse_softmax_cross_entropy_with_logits(
            labels=labels, logits=logit
        )
        return tf.reduce_sum(multi_ce_per_timestep)

    def _compute_pixel_control_loss(
        self, pc_net_q, pc_net_q_max, actions, pixel_change
    ):
        """
        pc_net_q: 像素控制输出的q
        pc_net_q_max: 像素控制输出的q_max
        actions: 智能体输出的动作 T*B*1
        pixel_change: 环境计算的像素变化 20*20
        """
        #######################
        # use all batch
        #######################
        # 临时变量存储计算的reward (TB) * 20 * 20
        pc_r = tf.Variable(
            tf.zeros((self.FLAGS.batch_size * (self.FLAGS.unroll_length - 1), 20, 20))
        )
        # 计算action矩阵
        pc_a = tf.one_hot(actions, len(DEFAULT_ACTION_SET))
        pc_a = pc_a[: self.FLAGS.unroll_length - 1]
        # 取出对应的Q值
        pc_q = pc_net_q[: self.FLAGS.unroll_length - 1]
        # 遍历计算新的r值，这里的r也按照强化学习的计算方式来计算，gamma加权的。
        for i in range(self.FLAGS.batch_size):
            # 倒序计算，首先赋最后一个值作为初始值。
            tf.assign(
                pc_r[i * self.FLAGS.batch_size + self.FLAGS.unroll_length - 2],
                pc_net_q_max[self.FLAGS.unroll_length - 1][i],
            )
            # 倒序，gamma加权计算r
            for j in range(self.FLAGS.unroll_length - 3, -1, -1):
                tf.assign(
                    pc_r[i * self.FLAGS.batch_size + j],
                    pixel_change[j][i]
                    + self.FLAGS.pc_gamma * pc_r[i * self.FLAGS.batch_size + j + 1],
                )

        # 计算Q的加权和，也就是v
        pc_a_reshape = tf.reshape(pc_a, [-1, 1, 1, len(DEFAULT_ACTION_SET)])
        pc_q = tf.reshape(pc_q, [-1, 20, 20, len(DEFAULT_ACTION_SET)])
        pc_qa = tf.multiply(pc_q, pc_a_reshape)
        pc_qa = tf.reduce_sum(pc_qa, reduction_indices=3, keep_dims=False)
        # 和原来的进行比较计算loss
        pc_loss = tf.nn.l2_loss(pc_r - pc_qa)

        return pc_loss

