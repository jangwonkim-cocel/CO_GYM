import copy


class Worker:
    def __init__(self, algorithm, env, rank, to_learner_queue, from_learner_queue, config):
        assert algorithm.type == 'off_policy'
        self.algorithm = copy.deepcopy(algorithm)
        self.env = copy.deepcopy(env)
        self.env.seed(rank)
        self.rank = rank
        self.config = config

        self.policy = self.algorithm.policy
        self.policy.to(config['worker_device'])

        self.buffer = self.algorithm.worker_buffer

        self.to_l_queue = to_learner_queue
        self.from_l_queue = from_learner_queue
        print(f"Load Worker (id: {rank})")

    def work(self):
        step = 0
        total_step = 0
        while True:
            epi_return = 0
            terminated, truncated = False, False
            state, _ = self.env.reset()
            while not (terminated or truncated):
                if total_step < (self.config['max_random_rollout'] / self.config['n_workers']):
                    action = self.env.random_action_sample()
                else:
                    action, _ = self.policy.get_action(state, eval=False, worker_device=self.config['worker_device'])

                next_states, reward, terminated, truncated, info = self.env.step(action)
                true_done = 0.0 if truncated else float(terminated or truncated)
                # assert self.algorithm.type == 'off_policy'
                self.buffer.push(state, action, reward, next_states, true_done)

                epi_return += reward
                step += 1
                total_step += 1
                state = next_states

                if step == self.config['max_rollout']:
                    worker_buffer = copy.deepcopy(self.buffer)
                    self.to_l_queue.put(worker_buffer)
                    self.policy = self.from_l_queue.get()
                    if self.config['worker_device'] == 'cuda':
                        self.policy.to('cuda')
                    self.buffer.clear()
                    step = 0

                if terminated or truncated:
                    break


