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

"""Functions to compute V-trace off-policy actor critic targets.

For details and theory see:

"IMPALA: Scalable Distributed Deep-RL with
Importance Weighted Actor-Learner Architectures"
by Espeholt, Soyer, Munos et al.

See https://arxiv.org/abs/1802.01561 for the full paper.
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

nest = tf.contrib.framework.nest

from tools import VTraceReturns, VTraceFromLogitsReturns

def log_probs_from_logits_and_actions(policy_logits, actions):
    """
    计算log pi
    """
    policy_logits = tf.convert_to_tensor(policy_logits, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    policy_logits.shape.assert_has_rank(3)
    actions.shape.assert_has_rank(2)
    # 计算log policy, log pi(a|x)
    # 给定的policy_logits是输出的动作概率，softmax之后计算ce = -log p(a|x), 取负之后得到log pi(a|x)
    return -tf.nn.sparse_softmax_cross_entropy_with_logits(
        logits=policy_logits, labels=actions
    )


def from_logits(
    behaviour_policy_logits,
    target_policy_logits,
    actions,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
    name="vtrace_from_logits",
):
    """
    V-trace for softmax policies.
    """

    behaviour_policy_logits = tf.convert_to_tensor(
        behaviour_policy_logits, dtype=tf.float32
    )
    target_policy_logits = tf.convert_to_tensor(target_policy_logits, dtype=tf.float32)
    actions = tf.convert_to_tensor(actions, dtype=tf.int32)

    # Make sure tensor ranks are as expected. The rest will be checked by from_action_log_probs.
    # behaviour, target: [T, B, actions]. actions: [T, B]
    behaviour_policy_logits.shape.assert_has_rank(3)
    target_policy_logits.shape.assert_has_rank(3)
    actions.shape.assert_has_rank(2)

    with tf.name_scope(
        name,
        values=[
            behaviour_policy_logits,
            target_policy_logits,
            actions,
            discounts,
            rewards,
            values,
            bootstrap_value,
        ],
    ):
        # 计算 softmax_cross_entropy
        target_action_log_probs = log_probs_from_logits_and_actions(
            target_policy_logits, actions
        )
        behaviour_action_log_probs = log_probs_from_logits_and_actions(
            behaviour_policy_logits, actions
        )
        log_rhos = target_action_log_probs - behaviour_action_log_probs

        vtrace_returns = from_importance_weights(
            log_rhos=log_rhos,
            discounts=discounts,
            rewards=rewards,
            values=values,
            bootstrap_value=bootstrap_value,
            clip_rho_threshold=clip_rho_threshold,
            clip_pg_rho_threshold=clip_pg_rho_threshold,
        )

        return VTraceFromLogitsReturns(
            log_rhos=log_rhos,
            behaviour_action_log_probs=behaviour_action_log_probs,
            target_action_log_probs=target_action_log_probs,
            **vtrace_returns._asdict()
        )


def from_importance_weights(
    log_rhos,
    discounts,
    rewards,
    values,
    bootstrap_value,
    clip_rho_threshold=1.0,
    clip_pg_rho_threshold=1.0,
    name="vtrace_from_importance_weights",
):
    """
    V-trace from log importance weights.
    """
    # covert to tensor
    log_rhos = tf.convert_to_tensor(log_rhos, dtype=tf.float32)
    discounts = tf.convert_to_tensor(discounts, dtype=tf.float32)
    rewards = tf.convert_to_tensor(rewards, dtype=tf.float32)
    values = tf.convert_to_tensor(values, dtype=tf.float32)
    bootstrap_value = tf.convert_to_tensor(bootstrap_value, dtype=tf.float32)

    if clip_rho_threshold is not None:
        clip_rho_threshold = tf.convert_to_tensor(clip_rho_threshold, dtype=tf.float32)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold = tf.convert_to_tensor(
            clip_pg_rho_threshold, dtype=tf.float32
        )

    # Make sure tensor ranks are consistent.
    rho_rank = log_rhos.shape.ndims  # Usually 2. [T, B]
    values.shape.assert_has_rank(rho_rank)  # [T, B]
    bootstrap_value.shape.assert_has_rank(rho_rank - 1)  # [B]
    discounts.shape.assert_has_rank(rho_rank)  # [T, B]
    rewards.shape.assert_has_rank(rho_rank)  # [T, B]

    if clip_rho_threshold is not None:
        clip_rho_threshold.shape.assert_has_rank(0)
    if clip_pg_rho_threshold is not None:
        clip_pg_rho_threshold.shape.assert_has_rank(0)

    with tf.name_scope(
        name, values=[log_rhos, discounts, rewards, values, bootstrap_value]
    ):
        """log_rhos = log(target_policy(a) / behaviour_policy(a)), 因此IS = e^log_rhos"""
        rhos = tf.exp(log_rhos)
        """rho = min(rho_bar, rho)"""

        cs = tf.minimum(1.0, rhos, name="cs")
        # Append bootstrapped value to get [v1, ..., v_t+1]
        values_t_plus_1 = tf.concat(
            [values[1:], tf.expand_dims(bootstrap_value, 0)], axis=0
        )
        """deltas_V = (r_t + gamma * V(x_(t+1)) - V(x_t))"""
        deltas = rewards + discounts * values_t_plus_1 - values

        sequences = (discounts, cs, deltas)

        """ 计算v_s - V(x_s)
        论文中的v_s可以递归的计算，v_s = V(x_s) + delta_t + discount * c_s * (v_s+1 - V(x_s+1))
        也就是说v_s - V(x_s) = delta_t + discount * c_s * (v_s+1 - V(x_s+1))
        继续可以得到v_s - V(x_s) = delta_t + discount * c_s * (v_s+1 - V(x_s+1))
                               = delta_t + discount * c_s * (delta_t + discount * c_s+1 * (v_s+2 - V(x_s+2)))
                               = ... + discount * c_s * (v_s+n - V(x_s+n)) = ... + discount * c_s * 0
        因此反向计算，初始值设置为0， 每次计算delta_t + discount * c_s * 上一次结果
        """

        def scanfunc(acc, sequence_item):
            discount_t, c_t, delta_t = sequence_item
            return delta_t + discount_t * c_t * acc

        initial_values = tf.zeros_like(bootstrap_value)
        # 见上面的说明
        vs_minus_v_xs = tf.scan(
            fn=scanfunc,
            elems=sequences,
            initializer=initial_values,
            parallel_iterations=1,
            back_prop=False,
            reverse=True,  # Computation starts from the back.
            name="scan",
        )

        # Add V(x_s) to get v_s.
        vs = tf.add(vs_minus_v_xs, values, name="vs")

        # Advantage for policy gradient.
        vs_t_plus_1 = tf.concat([vs[1:], tf.expand_dims(bootstrap_value, 0)], axis=0)
        # TODO 和论文存在区别,这里的是另一个参数，而不是论文里的复用rhos
        if clip_pg_rho_threshold is not None:
            clipped_pg_rhos = tf.minimum(
                clip_pg_rho_threshold, rhos, name="clipped_pg_rhos"
            )
        else:
            clipped_pg_rhos = rhos

        Lambda_1 = tf.constant(0.99, shape=cs.shape)
        # Lambda_2 = tf.constant(0.95, shape=cs.shape)
        delta_gae = rewards + discounts * values_t_plus_1 - values

        sequences_gae_1 = (discounts, Lambda_1, delta_gae)
        # sequences_gae_2 = (discounts, Lambda_2, delta_gae)

        initial_values = tf.zeros_like(bootstrap_value)
        pg_adv_1 = tf.scan(
            fn=scanfunc,
            elems=sequences_gae_1,
            initializer=initial_values,
            parallel_iterations=1,
            back_prop=False,
            reverse=True,  # Computation starts from the back.
            name="scan_gae_1",
        )

        # pg_advantages = (clipped_pg_rhos * (rewards + discounts * vs_t_plus_1 - values))
        pg_advantages = clipped_pg_rhos * tf.math.maximum(
            pg_adv_1, (rewards + discounts * vs_t_plus_1 - values)
        )

        # Make sure no gradients backpropagated through the returned values.
        return VTraceReturns(
            vs=tf.stop_gradient(vs), pg_advantages=tf.stop_gradient(pg_advantages)
        )

