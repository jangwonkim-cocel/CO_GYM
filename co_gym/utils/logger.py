import pickle
import os
import yaml
import torch


class OnPolicyLogger:
    def __init__(self, config):
        self.config = config
        self.checkpoint_no = 0
        if config['save_model']:
            self.save_dir = config['save_dir']
        else:
            self.save_dir = None

    def log(self, policy, critic, policy_optimizer_state_dict, critic_optimizer_state_dict, meta_data, normalizing_classes=None):
        assert self.save_dir is not None

        checkpoint_dir = self.save_dir + '/' + 'checkpoint_' + str(int(self.checkpoint_no + 1))
        os.mkdir(checkpoint_dir)
        with open(checkpoint_dir + '/metadata.yaml', 'w') as meta_f:
            yaml.dump(meta_data, meta_f)
        with open(checkpoint_dir + '/config.yaml', 'w') as config_f:
            yaml.dump(self.config, config_f)

        torch.save({'policy_state_dict': policy.state_dict(), 'critic_state_dict': critic.state_dict(),
                    'policy_optimizer_state_dict': policy_optimizer_state_dict,
                    'critic_optimizer_state_dict': critic_optimizer_state_dict}, checkpoint_dir + '/checkpoint.pt')
        torch.save(policy.state_dict(), checkpoint_dir + '/policy.pt')

        pickled_data_dir = checkpoint_dir + '/pickled_data/'
        # Save the normalizing information
        if normalizing_classes[0] is not None:
            obs_mean, obs_var = normalizing_classes[0].get()
            normalizing_data = {'mean': obs_mean.tolist(),
                                'std': (obs_var + 1e-6).tolist()}
            with open(checkpoint_dir + '/observation_normalizing_values.yaml', 'w') as norm_f:
                yaml.dump(normalizing_data, norm_f)

            if not os.path.exists(pickled_data_dir):
                os.mkdir(pickled_data_dir)

            with open(pickled_data_dir + 'ObsRunningMeanVar_class.pickle', 'wb') as p1:
                pickle.dump(normalizing_classes[0], p1)
            with open(pickled_data_dir + 'RetRunningMeanVar_class.pickle', 'wb') as p2:
                pickle.dump(normalizing_classes[1], p2)

        self.checkpoint_no += 1


class OffPolicyLogger:
    def __init__(self, config):
        self.config = config
        self.checkpoint_no = 0
        if config['save_model']:
            self.save_dir = config['save_dir']
        else:
            self.save_dir = None

    def log(self, policy, critic, policy_optimizer_state_dict, critic_optimizer_state_dict, meta_data,
            log_alpha, alpha_optimizer_state_dict, buffer):
        assert self.save_dir is not None

        checkpoint_dir = self.save_dir + '/' + 'checkpoint_' + str(int(self.checkpoint_no + 1))
        os.mkdir(checkpoint_dir)
        with open(checkpoint_dir + '/metadata.yaml', 'w') as meta_f:
            yaml.dump(meta_data, meta_f)
        with open(checkpoint_dir + '/config.yaml', 'w') as config_f:
            yaml.dump(self.config, config_f)

        '''
        torch.save({'policy_state_dict': policy.state_dict(), 'critic_state_dict': critic.state_dict(),
                    'policy_optimizer_state_dict': policy_optimizer_state_dict,
                    'critic_optimizer_state_dict': critic_optimizer_state_dict}, checkpoint_dir + '/checkpoint.pt')
        '''
        torch.save({'policy_state_dict': policy.state_dict(), 'critic_state_dict': critic.state_dict(),
                    'policy_optimizer_state_dict': policy_optimizer_state_dict,
                    'critic_optimizer_state_dict': critic_optimizer_state_dict,
                    'log_alpha': log_alpha, 'alpha_optimizer_state_dict': alpha_optimizer_state_dict,
                    'buffer': buffer},
                   checkpoint_dir + '/checkpoint.pt')

        torch.save(policy.state_dict(), checkpoint_dir + '/policy.pt')

        '''
        pickled_data_dir = checkpoint_dir + '/pickled_data/'

        with open(pickled_data_dir + 'buffer.pickle', 'wb') as p3:
            pickle.dump(buffer, p3)
        '''

        self.checkpoint_no += 1






