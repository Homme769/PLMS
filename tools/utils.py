import imp
import tensorflow as tf
import numpy as np
from tools import HUMAN_SCORES, RANDOM_SCORES, my_dm_lab


def fc_initializer(input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels)
        return tf.random_uniform(shape, minval=-d, maxval=d)

    return _initializer


def conv_initializer(kernel_width, kernel_height, input_channels, dtype=tf.float32):
    def _initializer(shape, dtype=dtype, partition_info=None):
        d = 1.0 / np.sqrt(input_channels * kernel_width * kernel_height)
        return tf.random_uniform(shape, minval=-d, maxval=d)

    return _initializer


def fc_variable(weight_shape, name):
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)

    input_channels = weight_shape[0]
    output_channels = weight_shape[1]
    bias_shape = [output_channels]

    weight = tf.get_variable(
        name_w, weight_shape, initializer=fc_initializer(input_channels)
    )
    bias = tf.get_variable(
        name_b, bias_shape, initializer=fc_initializer(input_channels)
    )
    return weight, bias


def conv_variable(weight_shape, name, deconv=False):
    """
    返回一个cov层的参数
    shape: [w, h, in, out]
    """
    name_w = "W_{0}".format(name)
    name_b = "b_{0}".format(name)

    w = weight_shape[0]
    h = weight_shape[1]
    if deconv:
        input_channels = weight_shape[3]
        output_channels = weight_shape[2]
    else:
        input_channels = weight_shape[2]
        output_channels = weight_shape[3]
    bias_shape = [output_channels]

    weight = tf.get_variable(
        name_w, weight_shape, initializer=conv_initializer(w, h, input_channels)
    )
    bias = tf.get_variable(
        name_b, bias_shape, initializer=conv_initializer(w, h, input_channels)
    )
    return weight, bias


def get2d_deconv_output_size(
    input_height, input_width, filter_height, filter_width, stride, padding_type
):
    """
    根据padding type 计算输出
    """
    if padding_type == "VALID":
        out_height = (input_height - 1) * stride + filter_height
        out_width = (input_width - 1) * stride + filter_width

    elif padding_type == "SAME":
        out_height = input_height * stride
        out_width = input_width * stride

    return out_height, out_width


def deconv2d(x, W, input_width, input_height, stride):
    """
    计算shape，调用网络de conv 2d
    """
    filter_height = W.get_shape()[0].value
    filter_width = W.get_shape()[1].value
    out_channel = W.get_shape()[2].value

    out_height, out_width = get2d_deconv_output_size(
        input_height, input_width, filter_height, filter_width, stride, "VALID"
    )
    batch_size = tf.shape(x)[0]
    output_shape = tf.stack([batch_size, out_height, out_width, out_channel])
    return tf.nn.conv2d_transpose(
        x, W, output_shape, strides=[1, stride, stride, 1], padding="VALID"
    )

# 计算像素的变化
def calc_pixel_change(last_state, state):
    # 除以255转为浮点数
    state = state.astype(np.float32) / 255.
    last_state = last_state.astype(np.float32) / 255.
    # 去除边框求差之后求平均数，转化为20*20
    d = np.absolute(state[2:-2, 2:-2, :] - last_state[2:-2, 2:-2, :])
    d = np.mean(d, 2)
    d = d.reshape([20, 4, 20, 4]).mean(-1).mean(1)
    return d

def _transform_level_returns(level_returns):
    """Converts training level names to test level names."""
    new_level_returns = {}
    for level_name, returns in level_returns.items():
        new_level_returns[my_dm_lab.get(level_name, level_name)] = returns

    test_set = set(my_dm_lab.values())
    diff = test_set - set(new_level_returns.keys())
    if diff:
        raise ValueError('Missing levels: %s' % list(diff))

    for level_name, returns in new_level_returns.items():
        if level_name in test_set:
            if not returns:
                raise ValueError('Missing returns for level: \'%s\': ' % level_name)
        else:
            tf.logging.info('Skipping level %s for calculation.', level_name)

    return new_level_returns


def compute_human_normalized_score(level_returns, per_level_cap):
    new_level_returns = _transform_level_returns(level_returns)

    def human_normalized_score(level_name, returns):
        score = np.mean(returns)
        human = HUMAN_SCORES[level_name]
        random = RANDOM_SCORES[level_name]
        human_normalized_score = (score - random) / (human - random) * 100
        if per_level_cap is not None:
            human_normalized_score = min(human_normalized_score, per_level_cap)
        return human_normalized_score

    return np.mean(
        [human_normalized_score(k, v) for k, v in new_level_returns.items()])