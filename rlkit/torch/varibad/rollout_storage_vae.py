import numpy as np
import torch


class RolloutStorageVAE(object):
    """Trajectory buffer for VariBAD's VAE.

    Completed BAMDP task-horizon trajectories are stored on CPU and sampled as
    [T, B, dim] tensors for the GRU encoder.
    """

    def __init__(self, num_processes, max_trajectory_len, max_num_rollouts,
                 state_dim, action_dim, vae_buffer_add_thresh=1.0):
        self.num_processes = num_processes
        self.max_traj_len = max_trajectory_len
        self.max_buffer_size = max_num_rollouts
        self.vae_buffer_add_thresh = vae_buffer_add_thresh
        self.insert_idx = 0
        self.buffer_len = 0

        self.prev_state = torch.zeros((self.max_traj_len, self.max_buffer_size, state_dim))
        self.next_state = torch.zeros((self.max_traj_len, self.max_buffer_size, state_dim))
        self.actions = torch.zeros((self.max_traj_len, self.max_buffer_size, action_dim))
        self.rewards = torch.zeros((self.max_traj_len, self.max_buffer_size, 1))
        self.trajectory_lens = [0] * self.max_buffer_size

        self.curr_timestep = torch.zeros((num_processes,), dtype=torch.long)
        self.running_prev_state = torch.zeros((self.max_traj_len, num_processes, state_dim))
        self.running_next_state = torch.zeros((self.max_traj_len, num_processes, state_dim))
        self.running_actions = torch.zeros((self.max_traj_len, num_processes, action_dim))
        self.running_rewards = torch.zeros((self.max_traj_len, num_processes, 1))

    def to_device(self, device):
        self.running_prev_state = self.running_prev_state.to(device)
        self.running_next_state = self.running_next_state.to(device)
        self.running_actions = self.running_actions.to(device)
        self.running_rewards = self.running_rewards.to(device)

    def get_running_batch(self):
        return (
            self.running_prev_state,
            self.running_next_state,
            self.running_actions,
            self.running_rewards,
            self.curr_timestep.clone(),
        )

    def insert(self, prev_state, actions, next_state, rewards, done):
        device = prev_state.device
        done = done.view(self.num_processes).bool()
        for i in range(self.num_processes):
            t = int(self.curr_timestep[i].item())
            if t >= self.max_traj_len:
                self._flush_running(i)
                t = 0

            self.running_prev_state[t, i].copy_(prev_state[i])
            self.running_next_state[t, i].copy_(next_state[i])
            self.running_actions[t, i].copy_(actions[i])
            self.running_rewards[t, i].copy_(rewards[i])
            self.curr_timestep[i] += 1

            if done[i]:
                self._flush_running(i)

        self.to_device(device)

    def _flush_running(self, process_idx):
        length = int(self.curr_timestep[process_idx].item())
        if length <= 0 or self.max_buffer_size <= 0:
            self._clear_running(process_idx)
            return
        if self.vae_buffer_add_thresh < np.random.uniform(0, 1):
            self._clear_running(process_idx)
            return

        if self.insert_idx >= self.max_buffer_size:
            self.insert_idx = 0
        self.prev_state[:, self.insert_idx] = self.running_prev_state[:, process_idx].detach().cpu()
        self.next_state[:, self.insert_idx] = self.running_next_state[:, process_idx].detach().cpu()
        self.actions[:, self.insert_idx] = self.running_actions[:, process_idx].detach().cpu()
        self.rewards[:, self.insert_idx] = self.running_rewards[:, process_idx].detach().cpu()
        self.trajectory_lens[self.insert_idx] = length
        self.insert_idx += 1
        self.buffer_len = min(max(self.buffer_len, self.insert_idx), self.max_buffer_size)
        self._clear_running(process_idx)

    def _clear_running(self, process_idx):
        self.running_prev_state[:, process_idx] *= 0
        self.running_next_state[:, process_idx] *= 0
        self.running_actions[:, process_idx] *= 0
        self.running_rewards[:, process_idx] *= 0
        self.curr_timestep[process_idx] = 0

    def ready_for_update(self):
        return self.buffer_len > 0

    def __len__(self):
        return self.buffer_len

    def get_batch(self, batchsize=5, replace=False, device=None):
        batchsize = min(self.buffer_len, batchsize)
        rollout_indices = np.random.choice(range(self.buffer_len), batchsize, replace=replace)
        trajectory_lens = np.array(self.trajectory_lens)[rollout_indices]
        device = device or self.running_prev_state.device
        return (
            self.prev_state[:, rollout_indices, :].to(device),
            self.next_state[:, rollout_indices, :].to(device),
            self.actions[:, rollout_indices, :].to(device),
            self.rewards[:, rollout_indices, :].to(device),
            trajectory_lens,
        )

