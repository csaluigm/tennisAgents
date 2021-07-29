import os
from agent import Agent
from config import Config
import random
from replay_buffer import ReplayBuffer
import torch

THIS_FOLDER = os.path.dirname(os.path.abspath(__file__))

print(torch.__version__)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(torch.cuda.is_available())

class MAgent():
    def __init__(self, agents, env, state_size, action_size, seed):
        self.gradient_clipping = True
        self.env = env
        self.agents = agents
        self.state_size = state_size * agents
        self.action_size = action_size * agents
        self.seed = random.seed(seed)
        self.config = Config()
        self.brain_name = env.brain_names[0]

        self.agent_left = Agent(self.state_size, self.action_size, self.config, device, seed=seed)
        self.agent_right = Agent( self.state_size, self.action_size, self.config, device, seed=seed, left_side=False)

        self.memory = ReplayBuffer(action_size, self.config.BUFFER_SIZE, self.config.BATCH_SIZE, seed, device)
        self.step_count = 0

    def step(self, states, actions, rewards, next_states, dones):
        self.memory.add(states.flatten(), actions.flatten(), rewards, next_states.flatten(), dones)
        self.step_count += 1

        if len(self.memory) > self.config.BATCH_SIZE and self.step_count % self.config.UPDATE_EVERY == 0:
            for _ in range(self.config.LEARN_TIMES):
                self.agent_left.learn( self.memory.sample(), self.agent_right)
                self.agent_right.learn( self.memory.sample(), self.agent_left)

    def save_model(self,prefix):
        torch.save(self.agent_left.actor_local.state_dict(),'left_actor_' + prefix +  '_trained_model.pth')
        torch.save(self.agent_right.actor_local.state_dict(),'right_actor_' + prefix +  '_trained_model.pth')
        torch.save(self.agent_left.critic_local.state_dict(),'left_critic_' + prefix +  '_trained_model.pth')
        torch.save(self.agent_right.critic_local.state_dict(),'right_critic_' + prefix +  '_trained_model.pth')

    def load_network_weights(self,model,path):
        model.load_state_dict(torch.load(os.path.join(THIS_FOLDER, path)))

    def load_model(self):
        self.load_network_weights(self.agent_left.critic_local,'left_critic_solved_trained_model.pth')    
        self.load_network_weights(self.agent_left.actor_local,'left_actor_solved_trained_model.pth')    
        self.load_network_weights(self.agent_right.critic_local,'right_critic_solved_trained_model.pth')    
        self.load_network_weights(self.agent_right.actor_local,'right_actor_solved_trained_model.pth')    
