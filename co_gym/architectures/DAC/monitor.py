import time
import copy
import wandb
import datetime
from co_gym.utils.logger import OffPolicyLogger


class Monitor:
    def __init__(self, env, from_learner_queue, total_epochs, config):
        self.env = copy.deepcopy(env)
        self.from_l_queue = from_learner_queue
        self.epochs = 0
        self.total_epochs = total_epochs
        self.logger = OffPolicyLogger(config)
        self.config = config
        print(f"Load Monitor")

    def evaluate(self, policy):
        if self.config['worker_device'] == 'cuda':
            policy.to('cuda')
        reward_list = []
        for epi_count in range(1, self.config['eval_episodes'] + 1):
            epi_reward = 0
            state, _ = self.env.reset()
            terminated = False
            truncated = False
            while not (terminated or truncated):
                action, _ = policy.get_action(state, eval=True, worker_device=self.config['worker_device'])
                try:
                    next_state, reward, terminated, truncated, info = self.env.step(action)
                except ValueError:
                    next_state, reward, terminated, truncated, info = self.env.step(action[0])

                state = next_state
                epi_reward += reward
            reward_list.append(epi_reward)

        avg_return = sum(reward_list) / len(reward_list)
        max_return = max(reward_list)
        min_return = min(reward_list)
        policy.to('cpu')
        return avg_return, max_return, min_return

    def monitoring(self):
        total_steps = 0
        start_time = time.time()
        timer_start = start_time

        if self.config['use_wandb']:
            wandb.login()
            wandb.init(project='co_gym', config=self.config)
            wandb.run.name = self.config['env_id'] + '/' + self.config['algorithm']
            wandb.define_metric("Wall Time (sec.)")
            wandb.define_metric("Timestep")
            wandb.define_metric("performance/Average return (w.r.t time)", step_metric="Wall Time (sec.)")
            wandb.define_metric("performance/Average return (w.r.t timestep)", step_metric="Timestep")
            wandb.define_metric("performance/Epochs", step_metric="Wall Time (sec.)")
            wandb.define_metric("performance/fps", step_metric="Wall Time (sec.)")

        while self.epochs < self.config['max_epochs']:
            self.epochs += 1

            if self.epochs % self.config['eval_freq'] == 0 and self.config['eval']:
                timer_end = time.time()
                plus_steps = self.config['eval_freq'] * self.config['max_rollout'] * self.config['n_workers']
                fps = plus_steps / (timer_end - timer_start)
                timer_start = timer_end
                wall_time = timer_end - start_time
                total_steps += plus_steps

                (policy, critic, policy_optimizer_state_dict, critic_optimizer_state_dict, log_alpha,
                 alpha_optimizer_state_dict, buffer) = self.from_l_queue.get()
                avg_return, max_return, min_return = self.evaluate(policy)
                print("Eval  |  Epochs [{}/{}]  |  Total Steps {}  |  fps {}  |  Average Return {:.2f}  |"
                      "  Max Return: {:.2f}  |  Min Return: {:.2f}  |  Wall-Time  {:.2f}".format(self.epochs,
                                                                                                 self.config['max_epochs'],
                                                                                                 total_steps, int(fps),
                                                                                                 float(avg_return),
                                                                                                 float(max_return),
                                                                                                 float(min_return),
                                                                                                 wall_time))
                if self.config['use_wandb']:
                    wandb.log({"performance/Average return (w.r.t time)": avg_return,
                               "performance/Average return (w.r.t timestep)": avg_return,
                               "performance/fps": fps,
                               "performance/Epochs": self.epochs,
                               "Wall Time (sec.)": wall_time,
                               "Timestep": total_steps})

                if self.config['save_model'] and self.epochs % self.config['model_checkpoint_freq'] == 0:
                    now = datetime.datetime.now()
                    cur_time = now.strftime('%Y-%m-%d_%H:%M:%S')
                    meta_data = {'epochs': self.epochs, 'average return': float(avg_return), 'timestep': total_steps,
                                 'wall time': wall_time, 'log_datetime': cur_time}

                    self.logger.log(policy, critic, policy_optimizer_state_dict, critic_optimizer_state_dict, meta_data,
                                    log_alpha, alpha_optimizer_state_dict, buffer)
                    print(f"Save the models ... checkpoint: {self.logger.checkpoint_no}")

            if self.epochs == self.config['max_epochs'] and self.config['use_wandb']:
                wandb.finish()
            with self.total_epochs.get_lock():
                self.total_epochs.value += 1

            time.sleep(0.005)
        return




