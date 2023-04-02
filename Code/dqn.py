"""
ECEN 743: Reinforcement Learning
Deep Q-Learning
Code tested using
	1. gymnasium 0.27.1
	2. box2d-py  2.3.5
	3. pytorch   2.0.0
	4. Python    3.9.12
1 & 2 can be installed using pip install gymnasium[box2d]

General Instructions
1. This code consists of TODO blocks, read them carefully and complete each of the blocks
2. Type your code between the following lines
			###### TYPE YOUR CODE HERE ######
			#################################
3. The default hyperparameters should be able to solve LunarLander-v2
4. You do not need to modify the rest of the code for this assignment, feel free to do so if needed.

"""
import gymnasium as gym
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import argparse
import numpy as np
from collections import deque, namedtuple

from utils import *


if __name__ == "__main__":
	parser = argparse.ArgumentParser()
	parser.add_argument("--env", default="LunarLander-v2")          # Gymnasium environment name
	parser.add_argument("--seed", default=0, type=int)              # sets Gym, PyTorch and Numpy seeds
	parser.add_argument("--n-episodes", default=2000, type=int)     # maximum number of training episodes
	parser.add_argument("--batch-size", default=64, type=int)       # training batch size
	parser.add_argument("--discount", default=0.99)                 # discount factor
	parser.add_argument("--lr", default=5e-4)                       # learning rate
	parser.add_argument("--tau", default=0.001)                     # soft update of target network
	parser.add_argument("--max-size", default=int(1e5),type=int)    # experience replay buffer length
	parser.add_argument("--update-freq", default=4, type=int)       # update frequency of target network
	parser.add_argument("--gpu-index", default=0,type=int)		    # GPU index
	parser.add_argument("--max-esp-len", default=1000, type=int)    # maximum time of an episode
	#exploration strategy
	parser.add_argument("--epsilon-start", default=1)               # start value of epsilon
	parser.add_argument("--epsilon-end", default=0.01)              # end value of epsilon
	parser.add_argument("--epsilon-decay", default=0.995)           # decay value of epsilon
	parser.add_argument("--render_mode", default = "rgb_array")
	args = parser.parse_args()

	# making the environment	
	env = gym.make(args.env, render_mode = args.render_mode)

	#setting seeds
	torch.manual_seed(args.seed)
	np.random.seed(args.seed)
	random.seed(args.seed)

	state_dim = env.observation_space.shape[0]
	action_dim = env.action_space.n

	kwargs = {
		"state_dim":state_dim,
		"action_dim":action_dim,
		"discount":args.discount,
	 	"tau":args.tau,
	 	"lr":args.lr,
	 	"update_freq":args.update_freq,
	 	"max_size":args.max_size,
	 	"batch_size":args.batch_size,
	 	"gpu_index":args.gpu_index
	}	
 
	#Begin Training
	learner = DQNAgent(**kwargs) #Creating the DQN learning agent
	windowSize = 100
	moving_window = deque(maxlen=windowSize)
	epsilon = args.epsilon_start #initial epsilon Value
	epsilonEnd = args.epsilon_end #Final Epsilon Value
	epsilonDecay = args.epsilon_decay #Epsilon Decay Rate
	scores = []
	for e in range(args.n_episodes):
		state, _ = env.reset(seed=args.seed)
		curr_reward = 0
		for t in range(args.max_esp_len):
			action = learner.select_action(state,epsilon) #To be implemented
			n_state,reward,terminated,truncated,_ = env.step(action)
			done = terminated or truncated 
			learner.step(state,action,reward,n_state,done) #To be implemented
			state = n_state
			curr_reward += reward
			if done:
				break
		moving_window.append(curr_reward)

		""""
		TODO: Write code for decaying the exploration rate using args.epsilon_decay
		and args.epsilon_end. Note that epsilon has been initialized to args.epsilon_start  
		1. You are encouraged to try new methods
		"""
		###### TYPE YOUR CODE HERE ######
		#################################	
		epsilon = max(epsilonEnd, epsilon*epsilonDecay)
		scores.append(np.mean(moving_window))
		if e % 100 == 0:
			print('Episode Number {} Average Episodic Reward (over 100 episodes): {:.2f}'.format(e, scores[-1]))
		
		#According to the documentation, we break when we are receiving a score of > 200 on average
		if scores[-1]>200:
			print("Converged in ", e, "episodes")
			break
	""""
	TODO: Write code for
	1. Logging and plotting
	2. Rendering the trained agent 
	"""
	###### TYPE YOUR CODE HERE ######
	#################################
	VisualizeData(learner, env, scores, windowSize, saveFig = True, suppress = False, **kwargs)
