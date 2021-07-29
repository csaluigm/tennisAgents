from ou_noise import OUNoise
import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import Adam
from model import Actor, Critic
class Agent:
    def __init__(self, state_size, action_size, config, device, seed=62900, left_side=True):

        self.left_side = left_side
        self.config = config
        self.seed = seed
        self.device = device
        self.action_size = action_size

        self.s_size = state_size // 2
        self.a_size = action_size // 2

        # Actor Networks 
        self.actor_local = Actor(state_size=self.s_size, action_size=self.a_size, seed=self.seed).to(device)
        self.actor_target = Actor(state_size=self.s_size, action_size=self.a_size, seed=self.seed).to(device)
        self.actor_optimizer = Adam( self.actor_local.parameters(), lr=self.config.LR_ACTOR)

        # Critic networks - combine both agents
        self.critic_local = Critic( state_size=state_size, action_size=action_size, seed=self.seed).to(device)
        self.critic_target = Critic( state_size=state_size, action_size=action_size, seed=self.seed).to(device)
        self.critic_optimizer = Adam( self.critic_local.parameters(), lr=self.config.LR_CRITIC, weight_decay=0)

        self.noise =  OUNoise(self.a_size, seed, config.ou_mu, config.ou_theta, config.ou_sigma)

    def act(self, state, add_noise=True, noise_value=None):
        state = torch.from_numpy(state).float().to(self.device)

        self.actor_local.eval()
        with torch.no_grad():
            action = self.actor_local(state)

        self.actor_local.train()

        if add_noise:
            action = action.cpu().data.numpy()
            action = np.clip(action + noise_value * self.noise.sample(), -1, 1)
        return action

    def filter_by_agent(self, values, size, current=True):
        if (self.left_side and current) or (not self.left_side and not current):
            return values[:, :size]
        else:
            return values[:, size:]

    def learn(self, experiences, opponent):
        states, actions, rewards, next_states, dones = experiences

        states_current =  self.filter_by_agent(states, self.s_size)

        self.filter_by_agent(actions, self.a_size),

        next_states_current =  self.filter_by_agent(next_states, self.s_size)

        states_opponent =  self.filter_by_agent(states, self.s_size, False)

        self.filter_by_agent(actions, self.a_size, False)

        next_states_opponent =  self.filter_by_agent(next_states, self.s_size, False)

        rewards =  rewards[:, int(not self.left_side)].reshape((-1, 1))
        dones =  dones[:, int(not self.left_side)].reshape((-1, 1))

        states = states
        actions = actions
        next_states = next_states

        # ---------------------------- update critic ---------------------------- #
        
        next_actions_current = self.actor_target(next_states_current)
        next_actions_opponent = opponent.actor_target(next_states_opponent)

        # combine the next actions from both agents
        to_concat = self.get_actions_order(next_actions_current,next_actions_opponent)
        actions_next = torch.cat(to_concat, dim=1).float().detach().to(self.device)

        # Get predicted next-state actions and Q values from target models
        Q_targets_next = self.critic_target(next_states, actions_next)
        # Compute Q targets for current states (y_i)
        Q_targets = rewards + self.config.GAMMA * Q_targets_next * (1 - dones)
        # Compute critic loss
        Q_expected = self.critic_local(states, actions)
        critic_loss = F.mse_loss(Q_expected, Q_targets.detach())
        # Minimize the loss
        self.critic_optimizer.zero_grad()
        critic_loss.backward()

        # use gradient clipping
        torch.nn.utils.clip_grad_norm_(self.critic_local.parameters(), 1)

        self.critic_optimizer.step()
        current_actions = self.actor_local(states_current)
        opponent_actions = opponent.actor_local(states_opponent)
        
        # ---------------------------- update actor ---------------------------- #
        actions_pred = torch.cat(self.get_actions_order(current_actions,opponent_actions),dim=1)
        # Minimize the loss
        self.actor_optimizer.zero_grad()
        actor_loss = - self.critic_local(states, actions_pred).mean()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.soft_update(self.critic_local, self.critic_target, self.config.TAU)
        self.soft_update(self.actor_local, self.actor_target, self.config.TAU)

    def get_actions_order(self,current,opponent):
        return [current,opponent] if self.left_side else [opponent,current]

    def soft_update(self, local_model, target_model, tau):
        for target_param, local_param in zip(target_model.parameters(), local_model.parameters()):
            target_param.data.copy_( tau*local_param.data + (1.0-tau)*target_param.data)