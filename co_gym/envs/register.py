from co_gym.envs.custom_envs.quad_inv_pendulum.quad_inv_pendulum_base import QuadInvPendulumBase
from co_gym.envs.custom_envs.quad_inv_pendulum.quad_inv_pendulum_env import QuadInvPendulumEnv
from co_gym.envs.custom_envs.quadrotor.quadrotor_base import QuadrotorBase
from co_gym.envs.custom_envs.quadrotor.quadrotor_env import QuadrotorEnv
from co_gym.envs.prebuilt_envs.openai_gym import OpenAIGym


# function make: Return the environment corresponding to the 'env_id'.
def make(env_id):
    # Prebuilt Environments (OpenAI Gym's MuJoCo)
    if env_id.startswith('Ant') or env_id.startswith('HalfCheetah') or env_id.startswith('Hopper') \
            or env_id.startswith('Humanoid') or env_id.startswith('Swimmer') or env_id.startswith('InvertedPendulum') \
            or env_id.startswith('Reacher') or env_id.startswith('Pusher'):
        return OpenAIGym(env_id=env_id)

    # Your Custom Environments (Put the environment into "co_gym.envs.custom_envs" first.)
    elif env_id == 'FlamingoStand-v0':
        pass
    elif env_id == 'Quadrotor-v0':
        return QuadrotorEnv(QuadrotorBase())
    elif env_id == 'QuadInvPendulum-v0':
        return QuadInvPendulumEnv(QuadInvPendulumBase())
    else:
        print("Register the environment first! (into: co_gym.envs.register.py)")
        raise NameError
