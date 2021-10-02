import collections

AgentOutput = collections.namedtuple(
    "AgentOutput", "action policy_logits baseline pc cate_task"
)

ActorOutput = collections.namedtuple(
    "ActorOutput", "level_name agent_state env_outputs agent_outputs"
)

PcOutput = collections.namedtuple("PcOutput", "pc_q, pc_max_q")

StepOutputInfo = collections.namedtuple('StepOutputInfo', 'episode_return episode_step')

StepOutput = collections.namedtuple('StepOutput', 'reward info done observation pixel task')

VTraceFromLogitsReturns = collections.namedtuple(
    'VTraceFromLogitsReturns',
    ['vs', 'pg_advantages', 'log_rhos', 'behaviour_action_log_probs', 'target_action_log_probs'])

VTraceReturns = collections.namedtuple('VTraceReturns', 'vs pg_advantages')

DEFAULT_ACTION_SET = (
    (0, 0, 0, 1, 0, 0, 0),  # Forward
    (0, 0, 0, -1, 0, 0, 0),  # Backward
    (0, 0, -1, 0, 0, 0, 0),  # Strafe Left
    (0, 0, 1, 0, 0, 0, 0),  # Strafe Right
    (-20, 0, 0, 0, 0, 0, 0),  # Look Left
    (20, 0, 0, 0, 0, 0, 0),  # Look Right
    (-20, 0, 0, 1, 0, 0, 0),  # Look Left + Forward
    (20, 0, 0, 1, 0, 0, 0),  # Look Right + Forward
    (0, 0, 0, 0, 1, 0, 0),  # Fire.
)

