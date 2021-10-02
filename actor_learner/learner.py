from .loss_fn import Loss
from .optimizer import Optimzer
from .retrace_v import from_logits
import tensorflow as tf

nest = tf.contrib.framework.nest

class Learner:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS
        self.loss = Loss(True, True, FLAGS.baseline_cost, 0, self.FLAGS)
        self.optimzer = Optimzer('rms', self.FLAGS)
    
    def __call__(self, agent, agent_state, env_outputs, agent_outputs):
        
        learner_outputs, _ = agent.unroll(agent_outputs.action, env_outputs, agent_state)

        # 最后一个值作为baseline的值
        bootstrap_value = learner_outputs.baseline[-1]

        """
        环境的输出是1 + FLAG.unroll个，将其输入到learner里面，获得的是对应每个时间t的时候learner的输出，
        同时我们获得了actor的输出也是1 + FLAG.unroll个，其中第t+1个是第t个时间环境的输出，因此对于从actor获得的经验，我们移除第一个初始化的情况
        对于learner的输出，我们移除最后一个，因为最后一个没有对应的actor的结果
        """
        agent_outputs = nest.map_structure(lambda t: t[1:], agent_outputs)
        rewards, infos, done, _, pixel_change, task_id = nest.map_structure(lambda t: t[1:], env_outputs)
        learner_outputs = nest.map_structure(lambda t: t[:-1], learner_outputs)

        # clip reward
        clipped_rewards = rewards
        if self.FLAGS.reward_clipping == 'abs_one':
            clipped_rewards = tf.clip_by_value(rewards, -1, 1)
        elif self.FLAGS.reward_clipping == 'soft_asymmetric':
            squeezed = tf.tanh(rewards / 5.0)
            # Negative rewards are given less weight than positive rewards.
            clipped_rewards = tf.where(rewards < 0, .3 * squeezed, squeezed) * 5.
        elif self.FLAGS.reward_clipping == 'none':
            clipped_rewards = rewards

        # cal discount
        discounts = tf.to_float(~done) * self.FLAGS.discounting

        # 数值校正
        # 返回 ['vs', 'pg_advantages', 'log_rhos', 'behaviour_action_log_probs', 'target_action_log_probs']
        with tf.device('/cpu'):
            vtrace_returns = from_logits(
                behaviour_policy_logits=agent_outputs.policy_logits,
                target_policy_logits=learner_outputs.policy_logits,
                actions=agent_outputs.action,
                discounts=discounts,
                rewards=clipped_rewards,
                values=learner_outputs.baseline,
                bootstrap_value=bootstrap_value)

        # 计算Loss
        num_env_frames = tf.train.get_global_step()
        classfy_rate = 0.5
        classfy_rate = tf.train.polynomial_decay(classfy_rate, num_env_frames, 1e7, 0)

        total_loss = self.loss(
            pg_advantages=vtrace_returns.pg_advantages,
            base_advantage=vtrace_returns.vs-learner_outputs.baseline,
            pg_logits=learner_outputs.policy_logits,
            cate_logits=learner_outputs.cate_task,
            actions=agent_outputs.action,
            cate_labels=task_id,
            cate_cost=classfy_rate,
            pix_cost=1,
            pc_net_q=learner_outputs.pc.pc_q, 
            pc_net_q_max=learner_outputs.pc.pc_max_q,
            pixel_change=pixel_change
        )

        # Optimization
        # 学习率逐渐的下降至0
        learning_rate = tf.train.polynomial_decay(self.FLAGS.learning_rate, num_env_frames, self.FLAGS.total_environment_frames, 0)
        optimizer = self.optimzer(learning_rate)
        # optimizer = tf.train.AdamOptimizer(learning_rate)
        train_op = optimizer.minimize(total_loss)

        # Merge updating the network and environment frames into a single tensor.
        # 更新全局的帧数
        with tf.control_dependencies([train_op]):
            num_env_frames_and_train = num_env_frames.assign_add(
                self.FLAGS.batch_size * self.FLAGS.unroll_length * self.FLAGS.num_action_repeats)

        # Adding a few summaries.
        tf.summary.scalar('learning_rate', learning_rate)
        tf.summary.scalar('total_loss', total_loss)
        tf.summary.histogram('action', agent_outputs.action)
        tf.summary.scalar('global_num_env_frames', num_env_frames)

        return done, infos, num_env_frames_and_train