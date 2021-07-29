from collections import deque
from cli import Cli
from config import Config
from  maddpg_agent import MAgent
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from unityagents import UnityEnvironment
import random
import sys

cli = Cli()
args = cli.parse()

env = UnityEnvironment(file_name="./Unity/Tennis.exe")

brain_name = env.brain_names[0]
brain = env.brains[brain_name]

env_info = env.reset(train_mode=True)[brain_name]

states = env_info.vector_observations
state_size = states.shape[1]
num_agents = len(env_info.agents)
action_size = brain.vector_action_space_size

seed = random.randint(0, int(1e3))

def train(episodes=2500):
    try:
        config = Config()
        model = MAgent(agents=num_agents,env=env,state_size=state_size, action_size=action_size, seed=seed)
        
        eps = config.EPS_START
        scores_window = deque(maxlen=100)
        scores = []

        for i_episode in range(1, episodes+1):
            env_info = env.reset(train_mode=False)[brain_name]
            states = env_info.vector_observations
            episode_score = np.zeros(num_agents)

            for _ in range(1000):
                states = np.reshape(states, (1, 48))
                
                state_left, state_right = split_by_size(states,24) 

                actions_left = model.agent_left.act( state_left, noise_value=eps)
                actions_right = model.agent_right.act( state_right, noise_value=eps)

                actions = np.vstack((actions_left.flatten(), actions_right.flatten()))
                brain_info = env.step(actions)[brain_name]

                next_states = brain_info.vector_observations
                rewards = brain_info.rewards
                dones = brain_info.local_done

                model.step(states,actions,rewards,next_states,dones)
                episode_score += rewards
                states = next_states

                eps = eps - config.LIN_EPS_DECAY 
                eps = np.maximum(eps, config.EPS_END)

                if np.any(dones):
                    break

            best_player_score = np.max(episode_score)
            scores_window.append(best_player_score)
            scores.append(best_player_score)

            log_step(i_episode,scores_window,eps,model)

        model.save_model('end')
        return scores
    except KeyboardInterrupt:
        model.save_model('interrupt')
        plot_score_chart(scores)
        sys.exit(0)

def test(episodes=100):
        model = MAgent(agents=num_agents,state_size=state_size, env=env, action_size=action_size, seed=seed)
        model.load_model()
        scores_window = deque(maxlen=100)

        for i_episode in range(1, episodes+1):
            env_info = env.reset(train_mode=False)[brain_name]
            states = env_info.vector_observations
            episode_score = np.zeros(num_agents)

            while True:
                states = np.reshape(states, (1, 48))
                state_left, state_right = split_by_size(states,24) 

                actions_left = model.agent_left.act( state_left, add_noise=False).detach().cpu().numpy()
                actions_right = model.agent_right.act( state_right, add_noise=False).detach().cpu().numpy()
                actions = np.vstack((actions_left.flatten(), actions_right.flatten()))

                brain_info = env.step(actions)[brain_name]

                next_states = brain_info.vector_observations
                rewards = brain_info.rewards
                dones = brain_info.local_done

                episode_score += rewards
                states = next_states

                if np.any(dones):
                    print("done")
                    break

            best_player_score = np.max(episode_score)
            scores_window.append(best_player_score)

            log_step(i_episode,scores_window,0,model)

def split_by_size(values,size):
   left = values[0,:size].reshape((1,size))
   right = values[0,size:].reshape((1,size)) 
   return left,right

def log_step(ep,scores_window,eps,model):
    
    print('\rEpisode {}\tAverage Score: {:.2f} Noise: {}'.format(ep, np.mean(scores_window), eps), end="")
    if ep % 2 == 0:
        print('\rEpisode {}\tAverage Score: {:.2f}'.format(ep, np.mean(scores_window)))

    if args.train:
        #calculate mean and check for solved model saving
        mean = np.mean(scores_window)
        if mean > 0.5 and mean <= 0.6:
            print('\nEnvironment solved in {:d} episodes!\tAverage Score: {:.2f}'.format(ep, np.mean(scores_window)))
            model.save_model('solved')

def plot_score_chart(scores):
    fig = plt.figure()
    fig.add_subplot(111)
    plt.plot(np.arange(len(scores)), scores)
    plt.title("Scores")
    rolling_mean = pd.Series(scores).rolling(100).mean()
    plt.plot(rolling_mean)
    plt.show()

if args.train:
    scores = train()
    plot_score_chart(scores)
else:
    scores = test()

    
# def print_env_info(env):
#     env_info = env.reset(train_mode=True)[brain_name]
#     # number of agents in the environment
#     print('Number of agents:', len(env_info.agents))
#     # number of actions
#     action_size = brain.vector_action_space_size
#     print('Number of actions:', action_size)
#     # examine the state space 
#     state = env_info.vector_observations[0]
#     print('States look like:', state)
#     state_size = len(state)
#     print('States have length:', state_size)
