import mujoco_py
import co_gym


def run_script():
    # Create Environment
    env_id = 'QuadInvPendulum-v0'
    #env_id ='Humanoid-v3'
    env = co_gym.make(env_id)

    # Set Algorithm & Configuration
    algorithm = 'TQC'
    c = {'n_workers': 32, 'worker_device': 'cpu', 'policy_hidden_dims': [64, 64], 'critic_hidden_dims': [64, 64], 'save_model': True, 'load_model': False, 'load_checkpoint_dir': 'PPO_QuadInvPendulum_2024-07-04_17:40:28/checkpoint_1'}
   
    # Set Trainer
    trainer = co_gym.load(env, algorithm=algorithm, config={'policy_hidden_dims': [512, 512, 512],
                                                            'critic_hidden_dims': [512, 512, 512, 512],
                                                            'load_model': True,
                                                            'load_checkpoint_dir': 'TQC_QuadInvPendulum-v0_2024-07-16_17:22:15/checkpoint_5'})
    trainer.train()

    # Set Tester
    tester = co_gym.Tester(env, algorithm=algorithm, checkpoint_dir='TQC_QuadInvPendulum-v0_2024-07-16_17:22:15/checkpoint_5')
    tester.test(eval_epi=50, render=True)


if __name__ == '__main__':
    run_script()


