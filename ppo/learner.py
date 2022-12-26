from typing import List

from ppo.actorcritic import Actor, Critic
from ppo.environment import Environment, CartPoleEnv, MountainEnv
import torch
import numpy as np
from torch.distributions import Categorical
import torch.nn as nn
import time
import os
import scipy.signal
from tqdm import tqdm


class Learner:
    """
    Class for the Reinforcement Learning Agent.
    """

    def __init__(self, state_dim, action_dim, lr_actor: float, lr_critic: float, eps_clip: float, size: float = 10000):
        """
        :param state_dim: number of features in state vector (int)
        :param action_dim: number of actions (for one-hot encoding)
        :param lr_actor: learning rate of actor (float)
        :param lr_critic: learning rate of critic (float)
        :param eps_clip: epsilon to use during clipping (for surrogate loss/objective)
        :param size: maximum size of buffer.
        NOTE: Size cannot be smaller than update_timestep, otherwise buffer would overflow.
        """
        ################################## set device ##################################
        print("============================================================================================")
        # set device to cpu or cuda
        device = torch.device('cpu')
        if torch.cuda.is_available():
            device = torch.device('cuda:0')
            torch.cuda.empty_cache()
            print("Device set to : " + str(torch.cuda.get_device_name(device)))
        else:
            print("Device set to : cpu")
        print("============================================================================================")
        self.device = device

        self.state_dim = state_dim
        self.action_dim = action_dim

        # Old / new actors and critics.
        # The old versions are used during the PPO weight updating scheme
        self.actor = Actor(state_dim=state_dim,
                           action_dim=action_dim).to(device)
        self.critic = Critic(state_dim=state_dim).to(device)

        self.old_actor = Actor(state_dim=state_dim,
                               action_dim=action_dim).to(device)
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic = Critic(state_dim=state_dim).to(device)
        self.old_critic.load_state_dict(self.critic.state_dict())

        # Optimizer for the two networks (the old networks will not be trained)
        self.optimizer = torch.optim.Adam([
            {'params': self.actor.parameters(), 'lr': lr_actor},
            {'params': self.critic.parameters(), 'lr': lr_critic}
        ])

        self.MseLoss = nn.MSELoss()  # MSE Loss

        self.eps_clip = eps_clip

        # "Rolling" buffer
        self.buffer = {'rewards': np.zeros(size, dtype=np.float32),
                       'is_terminals': np.zeros(size, dtype=np.float32),
                       'states': np.zeros((size, state_dim), dtype=np.float32),
                       'actions': np.zeros(size, dtype=np.float32),
                       'log_probs': np.zeros(size, dtype=np.float32),
                       'values': np.zeros(size, dtype=np.float32)}

        # Buffers for derived values (advantage, discounted rewards/returns)
        self.adv_buffer = np.zeros(size, dtype=np.float32)
        self.ret_buffer = np.zeros(size, dtype=np.float32)

        # Buffer state variables
        self.ptr, self.path_start_idx, self.max_size = 0, 0, size

    def store(self, reward, is_terminal, state, action, log_probs, value):
        """
        Store values in buffer, pointer is incremented.
        :param reward: scalar
        :param is_terminal: boolean (or 0/1 integer) indicating if terminal state or not
        :param state: list/numpy array representing state, assumed to be 1D
        :param action: integer indicating action
        :param log_probs: log probability for action
        :param value: value of state (expected future returns)
        :return: None
        """
        assert self.ptr < self.max_size, "Overflow in buffer, buffer must be flushed/cleared before storing more values"
        self.buffer['rewards'][self.ptr] = reward
        self.buffer['is_terminals'][self.ptr] = is_terminal
        self.buffer['states'][self.ptr] = state
        self.buffer['actions'][self.ptr] = action
        self.buffer['log_probs'][self.ptr] = log_probs
        self.buffer['values'][self.ptr] = value
        self.ptr += 1

    def finish_path(self, gamma, lamb, last_val: float = 0):
        """
        Calculate General Advantage Estimates and discounted rewards (returns) after a trajectory is finished.
        NOTE: Should be called before get_data() and clear_buffer().
        :param gamma: gamma value for GAE (trajectory discount factor)
        :param lamb: lambda value for GAE (exponential mean discount factor)
        :param last_val: last value (V(s_(T+1)), defaults to zero
        :return: None
        """
        path_slice = slice(self.path_start_idx, self.ptr)
        vals = np.append(self.buffer['values'][path_slice], last_val)
        rews = np.append(self.buffer['rewards'][path_slice], last_val)
        deltas = rews[:-1] + gamma * vals[1:] - vals[:-1]
        self.adv_buffer[path_slice] = self._discount_cumsum(deltas, gamma * lamb)
        self.ret_buffer[path_slice] = self._discount_cumsum(rews, gamma)[:-1]
        self.path_start_idx = self.ptr

    def get_data(self):
        """
        Get all the data from the running buffers (self.buffer[<key>]), and the data from the discounted rewards (returns)
        and advantages.
        NOTE: The data inn 'advantages' and 'discounted_rewards' is not calculated until finish_path() is called.
        Furthermore, get_data() should be called before clear_buffer() because pointers will be set to zero after
        clear_buffer() is invoked.
        :return: data in dictionary
        """
        data = {k: self.buffer[k][:self.ptr] for k in self.buffer}
        data['advantages'] = self.adv_buffer[:self.ptr]
        data['discounted_rewards'] = self.ret_buffer[:self.ptr]
        return data

    def load_buffer(self, results: List[np.ndarray]):
        keys = {'rewards',
                'is_terminals',
                'states',
                'actions',
                'log_probs',
                'values',
                'advantages',
                'discounted_rewards'}
        adv_buffer = None
        ret_buffer = None
        buffer = {}
        for result in results:
            if len(buffer.keys()) == 0:
                buffer = {'rewards': result['rewards'],
                          'is_terminals': result['is_terminals'],
                          'states': result['states'],
                          'actions': result['actions'],
                          'log_probs': result['log_probs'],
                          'values': result['values']}
                for key in buffer:
                    assert len(buffer[key]) <= self.max_size, "BUFFER IS OVERFLOWN"
                adv_buffer = result['advantages']
                assert len(adv_buffer) <= self.max_size, "BUFFER IS OVERFLOWN"
                ret_buffer = result['discounted_rewards']
                assert len(ret_buffer) <= self.max_size, "BUFFER IS OVERFLOWN"
                continue
            for key in set(result.keys()) & keys:
                if key == 'advantages':
                    adv_buffer = np.concatenate((adv_buffer, result[key]), axis=0)
                    assert len(adv_buffer) <= self.max_size, "BUFFER IS OVERFLOWN"
                    continue
                if key == 'discounted_rewards':
                    ret_buffer = np.concatenate((ret_buffer, result[key]), axis=0)
                    assert len(ret_buffer) <= self.max_size, "BUFFER IS OVERFLOWN"
                    continue
                buffer[key] = np.concatenate((buffer[key], result[key]), axis=0)
                assert len(buffer[key]) <= self.max_size, "BUFFER IS OVERFLOWN"
        self.buffer = buffer
        self.adv_buffer = adv_buffer
        self.ret_buffer = ret_buffer
        self.ptr = len(adv_buffer)

    def clear_buffer(self):
        """
        Clear buffer (that is, move pointer to current position and pointer to start of current path to 0).
        Should be called after updating the actor and critic networks (or straight after fetching the data using
        get_data())
        :return: None
        """
        self.ptr, self.path_start_idx = 0, 0

    def evaluate(self, state, action):
        """
        Evaluate (state, action) pair and record logarithmic probabilities from actor, state values from critic
        and the entropy of the probability distribution from the actor.
        NOTE: Use with torch.no_grad() or similarly if you do not want to compute gradients (not part of loss etc.)
        :param state: Torch tensor
        :param action: Torch tensor
        :return: log probabilities, state values, entropy of distribution <- Torch objects
        """
        action_probabilities = self.actor(state)
        distribution = Categorical(action_probabilities)
        action_log_probabilities = distribution.log_prob(action)
        distribution_entropy = distribution.entropy()
        state_values = self.critic(state)
        return action_log_probabilities, state_values, distribution_entropy

    @staticmethod
    def _discount_cumsum(x, discount):
        """
         magic from rllab for computing discounted cumulative sums of vectors.
         input:
             vector x,
             [x0,
              x1,
              x2]
         output:
             [x0 + discount * x1 + discount^2 * x2,
              x1 + discount * x2,
              x2]
         """
        return scipy.signal.lfilter([1], [1, float(-discount)], x[::-1], axis=0)[::-1]

    def generate_batches(self, discounted_rewards: np.ndarray, batch_size: int) -> np.ndarray:
        one_left = len(discounted_rewards) % batch_size == 1
        n_states = len(discounted_rewards)
        batch_start = np.arange(0, n_states, batch_size)
        indices = np.arange(n_states, dtype=np.int64)
        np.random.shuffle(indices)
        batches = [indices[i:i + batch_size] for i in batch_start if len(indices[i:i + batch_size]) > 1]
        if one_left:
            batches[-1] = np.concatenate((batches[-1], indices[-1:]))

        return batches

    # TODO: Incorporate batch size
    def update(self, epochs: int, batch_mode: bool = False, batch_size: int = 0) -> np.ndarray:
        """
        Update the weights for actor / critic network using PPO algorithm (clipped surrogate + MSE + Entropy bonus)
        Buffer is cleared after update is performed.
        :param batch_mode: True if batch mode is activated, otherwise False.
        :param epochs: number of epochs used for training
        :return: None
        """
        # Do not normalize discounted rewards (can be done by uncommenting below)
        data = self.get_data()
        discounted_rewards = data['discounted_rewards']
        batch_size = batch_size if batch_mode else len(discounted_rewards)
        batches = self.generate_batches(discounted_rewards, batch_size)

        # Training loop
        loss_dev = np.zeros(epochs)
        for e in range(epochs):
            for batch in batches:
                discounted_rewards = torch.tensor(data['discounted_rewards'][batch], dtype=torch.float32).to(
                    self.device)
                # discounted_rewards = (discounted_rewards - discounted_rewards.mean()) / (discounted_rewards.std() + 1e-7)

                # Normalize advantages
                advantages = torch.tensor(data['advantages'][batch], dtype=torch.float32).to(self.device)
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-7)

                # TODO: Should the states be represented as tensors rather than np arrays (speed improvement)
                # Convert data from buffer to torch tensors.
                old_states = torch.squeeze(
                    torch.stack(
                        [torch.tensor(state, dtype=torch.float32).to(self.device) for state in data['states'][batch]],
                        dim=0)).detach().to(self.device)
                old_actions = torch.squeeze(
                    torch.stack(
                        [torch.tensor(action, dtype=torch.float32).to(self.device) for action in
                         data['actions'][batch]],
                        dim=0)).detach().to(self.device)
                old_logprobs = torch.squeeze(
                    torch.stack(
                        [torch.tensor(log_prob, dtype=torch.float32).to(self.device) for log_prob in
                         data['log_probs'][batch]],
                        dim=0)).detach().to(self.device)

                # Get values from "newly trained" networks, these Tensors are the only ones
                # that are a part of the computational graph (others are "constants" in the loss functions)
                logprobs, state_values, dist_entropy = self.evaluate(old_states, old_actions)

                # Match state_values tensor dimensions with discounted_rewards tensor
                state_values = torch.squeeze(state_values)

                # Finding surrogate loss
                ratios = torch.exp(logprobs - old_logprobs.detach())
                surr1 = ratios * advantages
                surr2 = torch.clamp(ratios, 1 - self.eps_clip, 1 + self.eps_clip) * advantages

                # final loss of clipped objective PPO
                # -torch.min(surr1, surr2): Loss for actor
                # +MSELoss: Loss for critic
                # -dist_entropy: (Negative of) entropy bonus for actor
                loss = -torch.min(surr1, surr2) + 0.5 * self.MseLoss(state_values,
                                                                     discounted_rewards) - 0.01 * dist_entropy

                # Reset gradients of optimizer (to avoid accumulation), step backwards with loss function
                # (according to computational graph) and take a step with optimizer --> update parameters/weights.
                self.optimizer.zero_grad()
                loss.mean().backward()
                self.optimizer.step()
                loss_dev[e] += torch.sum(loss).item() / len(batch)

        # Copy new weights into old actor and critic
        self.old_actor.load_state_dict(self.actor.state_dict())
        self.old_critic.load_state_dict(self.critic.state_dict())

        # Clear buffer
        self.clear_buffer()
        return loss_dev

    def learn(self, env: Environment, epochs: int, gamma: float, lamb: float,
              max_episode_length: int, max_training_timesteps: int, log_freq: int, log_path: str, net_path: str,
              net_freq: int, update_timestep: int = 0, random_seed: int = 0):
        """
        Main training loop of the RL PPO Learner.
        :param env: Subclass of Environment class (must inherent and implement methods)
        :param epochs: number of epochs during each weight update (int)
        :param gamma: gamma value for GAE (trajectory discount factor)
        :param lamb: lambda value for GAE (exponential mean discount factor)
        :param max_episode_length: maximum number of steps in an episode (int)
        :param max_training_timesteps: maximum total steps during draining (int)
        :param log_freq: number of episodes between each log is written (int)
        :param log_path: path to the logfile (str)
        :param net_path: path to networks (actor/critic will be appended at end + count) (str)
        :param net_freq number of episodes between network parameters are saved (int)
        :param update_timestep: number of time steps between each update (defaults to max_episode_length * 4)
        :param random_seed: NO EFFECT YET.
        :return: None
        """
        assert self.state_dim == env.get_state_dim(), "State dimension of environment is not equal to state_dim"
        assert self.action_dim == env.get_action_dim(), "Action dimension is not equal to state_dim"

        time_steps = 0
        log_running_reward = 0
        log_running_episodes = 0

        net_running_episodes = 0

        episode_count = 0

        update_timestep = max_episode_length * 4 if update_timestep < 1 else update_timestep
        assert update_timestep <= self.max_size, "Cannot have update_timestep larger than max_size of buffer."

        log_f = open(log_path, "w+")
        log_f.write('episode,timestep,reward\n')

        # Continue until max training steps
        while time_steps < max_training_timesteps:
            print(f"{time_steps}/{max_training_timesteps}")
            state = env.reset()  # Reset environment
            current_episode_reward = 0

            if log_running_episodes >= log_freq:
                print("Writing to log...")
                log_avg_reward = log_running_reward / log_running_episodes
                log_avg_reward = round(log_avg_reward, 4)
                log_f.write(f'{episode_count},{time_steps},{log_avg_reward}\n')
                log_f.flush()
                log_running_episodes = 0
                log_running_reward = 0

            if net_running_episodes >= net_freq:
                print("Saving network parameters...")
                self.save_params(f'{net_path}actor{episode_count}', f'{net_path}critic{episode_count}')
                net_running_episodes = 0
            start = time.time()
            # Continue episode until max episode length is reached
            for t in tqdm(range(1, max_episode_length + 1)):

                # Do not compute gradients
                with torch.no_grad():
                    state = torch.FloatTensor(state).to(self.device)  # convert state FloatTensor
                    action, action_logprob = self.old_actor.act(state)  # Get action and log prob of action.
                    value = self.old_critic(state)  # Get critic evaluation of state

                action = action.item()  # Get numeric value (int) of action
                new_state, reward, done = env.step(action)  # Take a step in the environment, get information

                time_steps += 1
                current_episode_reward += reward

                timeout = t == max_episode_length  # Episode has reached max length
                update = time_steps % update_timestep == 0  # Should an update be performed?
                terminal = done or timeout or update  # Is this state a terminal (episode = finished)

                self.store(reward, done, state, action, action_logprob, value)
                if terminal:
                    print(t)
                    # If timeout: bootstrap with value estimate of new_state
                    # Else: value = 0
                    # TODO: Se på
                    if not done:
                        with torch.no_grad():
                            value = self.old_critic(torch.FloatTensor(new_state).to(self.device))
                    else:
                        value = 0
                    # Finish the path (calculate advantages + discounted rewards)
                    self.finish_path(gamma, lamb, value)
                    if update:
                        print("Updating...")
                        self.update(epochs)  # Update actor and critic
                    log_running_reward += current_episode_reward
                    log_running_episodes += 1
                    net_running_episodes += 1
                    episode_count += 1
                    print(f"Time serialized: {time.time() - start}")
                    break

                state = new_state
        log_f.close()

    def test(self, env: Environment, max_episode_length: int, visualize: bool = False, delay: float = 0.01):
        """
        Test the Learner agent.
        :param env: Subclass of Environment class (must inherent and implement methods)
        :param max_episode_length: maximum number of steps in an episode (int)
        :param visualize: True if the run should be visualized, False otherwise.
        :param delay: number of seconds to wait during each frame rendering during visualization (float)
        :return: None
        Note: render() method must be implemented for visualization to work.
        """
        assert self.state_dim == env.get_state_dim(), "State dimension of environment is not equal to state_dim"
        assert self.action_dim == env.get_action_dim(), "Action dimension is not equal to state_dim"

        state = env.reset()
        current_episode_reward = 0
        if visualize:
            env.render()
            time.sleep(delay)
        for t in range(1, max_episode_length + 1):
            with torch.no_grad():
                state = torch.FloatTensor(state).to(self.device)
                action, action_logprob = self.old_actor.act(state)
            action = action.item()
            state, reward, done = env.step(action)
            done = True if t == max_episode_length else done

            if visualize:
                env.render()
                time.sleep(delay)

            current_episode_reward += reward

            if done:
                print(t)
                break

    def save_params(self, path1, path2):
        torch.save(self.old_actor.state_dict(), path1)
        torch.save(self.old_critic.state_dict(), path2)

    def load_params(self, path1, path2):
        actor_dict = torch.load(path1)
        self.actor.load_state_dict(actor_dict)
        self.old_actor.load_state_dict(actor_dict)

        critic_dict = torch.load(path2)
        self.critic.load_state_dict(critic_dict)
        self.old_critic.load_state_dict(critic_dict)


if __name__ == "__main__":
    pass