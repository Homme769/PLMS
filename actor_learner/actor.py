import tensorflow as tf
from tools import ActorOutput

nest = tf.contrib.framework.nest


class Actor:
    def __init__(self, FLAGS):
        self.FLAGS = FLAGS

    def __call__(self, agent, env, level_name, action_set):
        """ Actor 行动者，负责采集数据
        :param agent: 学习的网络
        :param env: 可以一直交互的环境，经过封装的结束时自动重新开始的环境FlowEnvironment
        :param level_name: 关卡名字
        :param action_set: 有效的动作集，用来进行动作抽样生成
        :rtype: 返回ActorOutput类型的数据，见tools/define.py
        """
        
        """获得初始值，用以后面的初始化"""
        initial_env_output, initial_env_state = env.initial()
        initial_agent_state = agent.initial_state(1)
        initial_action = tf.zeros([1], dtype=tf.int32)
        """
        在snt 1.23中__call__函数会自动调用_build函数，这里等价于agent._build(...)
        初始状态输入，获得agent的初始输出结构
        """
        dummy_agent_output, _ = agent(
            (
                initial_action,
                nest.map_structure(lambda t: tf.expand_dims(t, 0), initial_env_output), # 将每一个的第1维扩展
            ),
            initial_agent_state,
        )
        """根据结构初始化为0"""
        initial_agent_output = nest.map_structure(
            lambda t: tf.zeros(t.shape, t.dtype), dummy_agent_output
        )

        # 以上的初始化值中，除了环境的output中obs以外，其余全部为对应的结构，其值为0

        # 需要跨iter保留的数据，例如环境的状态，环境的输出，智能体的状态和输出
        # 也就是将这些数据包装为local_variable
        def create_state(t):
            # 如果有，则获取。没有则生成
            with tf.variable_scope(None, default_name="state"):
                return tf.get_local_variable(
                    t.op.name, initializer=t, use_resource=True
                )

        persistent_state = nest.map_structure(
            create_state,
            (
                initial_env_state,
                initial_env_output,
                initial_agent_state,
                initial_agent_output,
            ),
        )

        def step(input_, unused_i):
            # 每次获取对应的状态，执行智能体，获取新的动作，之后继续，这里不断的采集数据
            # 因为环境已经保证如果游戏结束的话，自动开始下一局游戏，因此可以不间断的采集
            env_state, env_output, agent_state, agent_output = input_

            action = agent_output[0]
            batched_env_output = nest.map_structure(
                lambda t: tf.expand_dims(t, 0), env_output
            )
            agent_output, agent_state = agent((action, batched_env_output), agent_state)

            action = agent_output[0][0]
            raw_action = tf.gather(action_set, action)

            env_output, env_state = env.step(raw_action, env_state)

            return env_state, env_output, agent_state, agent_output

        # Run the unroll. `read_value()` is needed to make sure later usage will
        # return the first values and not a new snapshot of the variables.
        first_values = nest.map_structure(lambda v: v.read_value(), persistent_state)
        _, first_env_output, first_agent_state, first_agent_output = first_values

        # 使用scan函数循环的执行step函数FLAGS.unroll_length次，初始值为first_values
        # 最终每一次的返回都会在output中
        output = tf.scan(step, tf.range(self.FLAGS.unroll_length), first_values)
        _, env_outputs, _, agent_outputs = output

        # 更新local variable
        assign_ops = nest.map_structure(
            lambda v, t: v.assign(t[-1]), persistent_state, output
        )

        # The control dependency ensures that the final agent and environment states
        # and outputs are stored in `persistent_state` (to initialize next unroll).
        with tf.control_dependencies(nest.flatten(assign_ops)):
            # Remove the batch dimension from the agent state/output.
            first_agent_state = nest.map_structure(lambda t: t[0], first_agent_state)
            first_agent_output = nest.map_structure(lambda t: t[0], first_agent_output)
            agent_outputs = nest.map_structure(lambda t: t[:, 0], agent_outputs)

            # Concatenate first output and the unroll along the time dimension.
            # 将初始状态加入step列表中
            full_agent_outputs, full_env_outputs = nest.map_structure(
                lambda first, rest: tf.concat([[first], rest], 0),
                (first_agent_output, first_env_output),
                (agent_outputs, env_outputs),
            )

            output = ActorOutput(
                level_name=level_name,
                agent_state=first_agent_state,
                env_outputs=full_env_outputs,
                agent_outputs=full_agent_outputs,
            )

            # 这里只是经验，不需要进行梯度
            return nest.map_structure(tf.stop_gradient, output)
