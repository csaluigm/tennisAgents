# DDPG Agent
MADDPG for the Unity Tennis environment

![](https://media.giphy.com/media/ecHjsqwZQWPA4VoM4D/giphy.gif)
# Project Details
In this environment, two agents control rackets to bounce a ball over a net. If an agent hits the ball over the net, it receives a reward of +0.1. If an agent lets a ball hit the ground or hits the ball out of bounds, it receives a reward of -0.01. Thus, the goal of each agent is to keep the ball in play.

The observation space consists of 8 variables corresponding to the position and velocity of the ball and racket. Each agent receives its own, local observation. Two continuous actions are available, corresponding to movement toward (or away from) the net, and jumping.

The task is episodic, and in order to solve the environment, your agents must get an average score of +0.5 (over 100 consecutive episodes, after taking the maximum over both agents). Specifically,

After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 2 (potentially different) scores. We then take the maximum of these 2 scores.
This yields a single score for each episode.
The environment is considered solved, when the average (over 100 episodes) of those scores is at least +0.5.
# Getting Started

Make sure you have python installed in your computer. (The project was created with python 3.6.12) [download python](https://www.python.org/downloads/)

Navigate to the root of the project:

`cd Tennis` 

Install required python packages using pip or conda, for a quick basic setup use:

`pip install -r requirements.txt` 

The repo already contains the windowsx64 version of the unity environment otherwise you would need to download it and place it under the Unity folder.

* Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Linux.zip)
* Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis.app.zip)
* Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86.zip)
* Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P3/Tennis/Tennis_Windows_x86_64.zip)

# Instructions

You can run the project from some Editor like VS code or directly from commandline:

`python main.py --train 1`

This will train the agent and will store 2 versions of model weights. One when it pass the environment solved condition and other after the training episodes.

it stores the model in actor_trained_model.pth and critic_trained_model.pth on the root of the project.

Once the model is trained you can check its behavior by testing it:

`python main.py`