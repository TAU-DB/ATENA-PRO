import logging
from gym.envs.registration import register

logger = logging.getLogger(__name__)

register(
    id='ATENAld-v0',
    entry_point='gym_ianna.envs:ATENAEnv',
#    timestep_limit=1000,
#    reward_threshold=1.0,
#    nondeterministic = True,
)

register(
    id='ATENAPROcont-v0',
    entry_point='gym_atena_pro.envs.atena_pro_env_cont:ATENAPROEnvCont',
)