import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import csv
from os import mkdir, path

# Define the Actor and Critic networks
class Actor(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Actor, self).__init__()
        self.fc1 = nn.Linear(state_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, action_dim)

    def forward(self, state):
        x = torch.relu(self.fc1(state))
        x = torch.relu(self.fc2(x))
        action = torch.tanh(self.fc3(x))
        return action

class Critic(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim):
        super(Critic, self).__init__()
        self.fc1 = nn.Linear(state_dim + action_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, 1)

    def forward(self, state, action):
        x = torch.cat([state, action], 1)
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        q_value = self.fc3(x)
        return q_value

# DDPG Agent
class DDPGAgent:
    def __init__(self, state_dim, action_dim, hidden_dim, replay_buffer, max_action):
        self.actor = Actor(state_dim, action_dim, hidden_dim)
        self.critic = Critic(state_dim, action_dim, hidden_dim)
        self.target_actor = Actor(state_dim, action_dim, hidden_dim)
        self.target_critic = Critic(state_dim, action_dim, hidden_dim)

        self.replay_buffer = replay_buffer
        self.max_action = max_action

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=1e-4)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=1e-3)

        self.update_target_networks(tau=1)

    def save_weights(self, best=True):
        try:
            mkdir('weights')
        except:
            pass
        try:
            mkdir('weights/best')
        except:
            pass
        try:
            mkdir('weights/last')
        except:
            pass

        if best:
            base = "weights/best"
        else:
            base = "weights/last"

        torch.save(self.actor.state_dict(), path.join(base,'actor.pth'))
        torch.save(self.target_actor.state_dict(), path.join(base,'target_actor.pth'))
        torch.save(self.critic.state_dict(), path.join(base,'critic.pth'))
        torch.save(self.target_critic.state_dict(), path.join(base,'target_critic.pth'))

    def load_weights(self, best=True):
        if best:
            base = "weights/best"
        else:
            base = "weights/last"

        self.actor.load_state_dict(torch.load(path.join(base,'actor.pth')))
        self.target_actor.load_state_dict(torch.load(path.join(base,'target_actor.pth')))
        self.critic.load_state_dict(torch.load(path.join(base,'critic.pth')))
        self.target_critic.load_state_dict(torch.load(path.join(base,'target_critic.pth')))

    def update_target_networks(self, tau=0.005):
        critic_state_dict = self.critic.state_dict()
        actor_state_dict = self.actor.state_dict()
        for key in self.target_critic.state_dict():
            self.target_critic.state_dict()[key] = tau * critic_state_dict[key] + (1 - tau) * self.target_critic.state_dict()[key]
        for key in self.target_actor.state_dict():
            self.target_actor.state_dict()[key] = tau * actor_state_dict[key] + (1 - tau) * self.target_actor.state_dict()[key]

    def get_action(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0)
        action = self.actor(state).squeeze(0).detach().numpy()
        return self.clip_action(action)
    
    def clip_action(self, action):
        for i in range(len(action)):
            action[i] = np.clip(action[i], -self.max_action[i], self.max_action[i])
        return action


    def train(self, batch_size):
        states, actions, rewards, next_states, dones = self.replay_buffer.sample(batch_size)

        states = torch.from_numpy(states).float()
        actions = torch.from_numpy(actions).float()
        rewards = torch.from_numpy(rewards).float().unsqueeze(1)
        next_states = torch.from_numpy(next_states).float()
        dones = torch.from_numpy(dones).float().unsqueeze(1)

        # Update Critic
        next_actions = self.target_actor(next_states)
        next_q_values = self.target_critic(next_states, next_actions).detach()
        q_targets = rewards + (1 - dones) * 0.99 * next_q_values
        q_values = self.critic(states, actions)
        critic_loss = nn.MSELoss()(q_values, q_targets)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Update Actor
        actor_loss = -self.critic(states, self.actor(states)).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_networks()


class ReplayBuffer:
    def __init__(self, buffer_size, state_dim, action_dim):
        self.buffer_size = buffer_size
        self.state_buffer = np.zeros((self.buffer_size, state_dim))
        self.action_buffer = np.zeros((self.buffer_size, action_dim))
        self.reward_buffer = np.zeros(self.buffer_size)
        self.next_state_buffer = np.zeros((self.buffer_size, state_dim))
        self.done_buffer = np.zeros(self.buffer_size, dtype=bool)
        
        self.ptr = 0
        self.size = 0

    def add(self, state, action, reward, next_state, done):
        self.state_buffer[self.ptr] = state
        self.action_buffer[self.ptr] = action
        self.reward_buffer[self.ptr] = reward
        self.next_state_buffer[self.ptr] = next_state
        self.done_buffer[self.ptr] = done

        self.ptr = (self.ptr + 1) % self.buffer_size
        self.size = min(self.size + 1, self.buffer_size)

    def sample(self, batch_size):
        indices = np.random.choice(self.size, batch_size, replace=False)

        states = self.state_buffer[indices]
        actions = self.action_buffer[indices]
        rewards = self.reward_buffer[indices]
        next_states = self.next_state_buffer[indices]
        dones = self.done_buffer[indices]

        return states, actions, rewards, next_states, dones

    def save_as_csv(self, file_path):
        with open(file_path, 'w', newline='') as csvfile:
            writer = csv.writer(csvfile)
            writer.writerow(['state', 'action', 'reward', 'next_state', 'done'])
            for i in range(self.size):
                writer.writerow([
                    ','.join(map(str, self.state_buffer[i])),
                    ','.join(map(str, self.action_buffer[i])),
                    str(self.reward_buffer[i]),
                    ','.join(map(str, self.next_state_buffer[i])),
                    str(int(self.done_buffer[i]))
                ])

    def load_from_csv(self, file_path):
        with open(file_path, 'r') as csvfile:
            reader = csv.reader(csvfile)
            next(reader)  # Skip header row
            for row in reader:
                state = np.array([float(x) for x in row[0].split(',')])
                action = np.array([float(x) for x in row[1].split(',')])
                reward = float(row[2])
                next_state = np.array([float(x) for x in row[3].split(',')])
                done = bool(int(row[4]))

                self.add(state, action, reward, next_state, done)

class EpsilonGreedy:
    def __init__(self, start_eps, end_eps, eps_decay, max_steps, sigma):
        self.start_eps = start_eps
        self.end_eps = end_eps
        self.eps_decay = eps_decay
        self.max_steps = max_steps
        self.sigma = sigma
        self.current_step = 0
        self.epsilon = start_eps

    def get_epsilon(self):
        if self.current_step >= self.max_steps:
            return self.end_eps
        else:
            self.epsilon = self.start_eps * np.exp(-self.current_step * self.eps_decay)
            return self.epsilon
    
    def incr_step(self):
        self.current_step += 1

    def reset(self):
        self.current_step = 0
        self.epsilon = self.start_eps

    def get_action(self, agent, state):
        if np.random.rand() < self.get_epsilon():
            # Explore: Take a random action
            action = np.asarray([1, np.random.normal(0, self.sigma)]) 
            # print(action)
            action = agent.clip_action(action)
        else:
            # Exploit: Take the action recommended by the agent
            action = agent.get_action(state)

        return action