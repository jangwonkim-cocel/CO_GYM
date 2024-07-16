from co_gym.algorithms.ddpg import DDPG
from co_gym.algorithms.ppg import PPG
from co_gym.algorithms.ppo import PPO
from co_gym.algorithms.sac import SAC
from co_gym.algorithms.tqc import TQC
from co_gym.algorithms.td3 import TD3
from co_gym.architectures.D2AC.d2ac import D2AC
from co_gym.architectures.DAC.dac import DAC
from co_gym.envs.wrappers.normalize import NormalizedEnv
import pickle
import torch
import os
import datetime
import inspect
import yaml


# function load: Return a trainer (object) loaded with the corresponding [environment, algorithm, configurations].
def load(env, algorithm, config=None):
    return Trainer(env, algorithm, config)


class Trainer:
    def __init__(self, env, algorithm, config):
        print('*** Co-Gym (version 0.2.1)***\n')

        abs_path = inspect.getfile(self.__class__)[:-10]
        if algorithm in ['ppo', 'ppg', 'PPO', 'PPG']:
            with open(abs_path + 'config/on_policy_default.yaml') as on_policy_yaml_f:
                default_config = yaml.load(on_policy_yaml_f, Loader=yaml.FullLoader)
        elif algorithm in ['ddpg', 'td3', 'sac', 'tqc', 'DDPG', 'TD3', 'SAC', 'TQC']:
            with open(abs_path + 'config/off_policy_default.yaml') as off_policy_yaml_f:
                default_config = yaml.load(off_policy_yaml_f, Loader=yaml.FullLoader)
        else:
            print("Choose an algorithm in {'PPO', 'PPG', 'DDPG', 'TD3', 'SAC', 'TQC'}.")
            raise NameError
        default_config['env_id'] = env.id
        default_config['algorithm'] = algorithm

        # If the user hasn't provided a specific configuration, the default configuration is used.
        if config is None:
            config = default_config
        # If the user has provided, check
        else:
            try:
                keys = config.keys()
            except AttributeError:
                print(f"The configuration must be in Python's Dictionary form. But your type of configuration is"
                      f" {type(config)}.")
                raise

            for key in keys:
                if key not in default_config.keys():
                    print(f"Wrong configuration! There is no such configuration ({key}) in algorithm {algorithm}.")
                    raise KeyError
                else:
                    default_config[key] = config[key]
            config = default_config

        # Wrap the Environment
        try:
            if config['normalize_obs'] or config['normalize_reward']:
                env = NormalizedEnv(env)
        except KeyError:
            pass
        except AttributeError:
            pass

        # Save the model?
        if config['save_model']:
            now = datetime.datetime.now()
            cur_time = now.strftime('%Y-%m-%d_%H:%M:%S')
            save_dir = abs_path + 'log/' + config['algorithm'] + '_' + config['env_id'] + '_' + cur_time
            config['save_dir'] = save_dir
            try:
                os.mkdir(save_dir)
            except:
                print(f"Can't make the log-folder at: {save_dir}")
                raise OSError
            print(f'Trained model will be saved at: {save_dir}')

        # Use Weight & Bias (wandb)?
        if config['use_wandb'] is False:
            print("To enable tracking, you can set 'use_wandb': True (currently set to False).")
        print()
        # Set the Algorithm
        if algorithm in ['ppo', 'PPO']:
            algorithm = PPO(env, config)
        elif algorithm in ['ppg', 'PPG']:
            algorithm = PPG(env, config)
        elif algorithm in ['ddpg', 'DDPG']:
            algorithm = DDPG(env, config)
        elif algorithm in ['td3', 'TD3']:
            algorithm = TD3(env, config)
        elif algorithm in ['sac', 'SAC']:
            algorithm = SAC(env, config)
        elif algorithm in ['tqc', 'TQC']:
            algorithm = TQC(env, config)
        else:
            print("Choose an algorithm in {'PPO', 'PPG', 'DDPG', 'TD3', 'SAC', 'TQC'}.")
            raise NameError

        # Load Saved model?
        if config['load_model']:
            checkpoint_dir = abs_path + 'log/' + config['load_checkpoint_dir'] + '/checkpoint.pt'
            print(f"\nLoad the saved model from: {checkpoint_dir}\n")
            checkpoint = torch.load(checkpoint_dir)
            algorithm.policy.load_state_dict(checkpoint['policy_state_dict'])
            algorithm.critic.load_state_dict(checkpoint['critic_state_dict'])

            if algorithm.type == 'on_policy':
                pickled_data_dir = abs_path + 'log/' + config['load_checkpoint_dir']
                if config['normalize_obs'] or config['normalize_reward']:
                    with open(pickled_data_dir + '/pickled_data/ObsRunningMeanVar_class.pickle', 'rb') as p1:
                        obs_mean_var = pickle.load(p1)
                    with open(pickled_data_dir + '/pickled_data/RetRunningMeanVar_class.pickle', 'rb') as p2:
                        ret_mean_var = pickle.load(p2)

                    obs_mean, obs_var = obs_mean_var.get()
                    _, ret_var = ret_mean_var.get()
                    env.set_obs_mean_var(obs_mean, obs_var)
                    env.set_ret_var(ret_var)

        # Set the Architecture
        if algorithm.type == 'on_policy':
            self.architecture = D2AC(env, algorithm, config)  # Dual Distributed Actor-Critic (D2A2C).
        elif algorithm.type == 'off_policy':
            self.architecture = DAC(env, algorithm, config)   # Distributed Actor-Critic (DA2C).
        else:
            print("Algorithm's type must be either 'on_policy' or 'off_policy'.")
            raise NameError

        if config['save_model']:
            assert config['model_checkpoint_freq'] >= config['eval_freq']
            assert config['model_checkpoint_freq'] % config['eval_freq'] == 0

        self.config = config
        self.algorithm = algorithm
        self.env = env

    def train(self):
        self.architecture.train()
        return

