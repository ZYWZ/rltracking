from gym.envs.registration import register

register(
    id='rltracking-v0',
    entry_point='gym_rltracking.envs:RltrackingEnv',
)

register(
    id='rltracking-v1',
    entry_point='gym_rltracking.envs:GymRltrackingEnv',
)
