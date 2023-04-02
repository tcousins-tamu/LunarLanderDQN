"""
This file contains the classes and functions used by the Lunar Lander Implementation

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
#For visualization
import matplotlib.pyplot as plt
import glob
from gymnasium.wrappers.monitoring import video_recorder
from IPython.display import HTML
from IPython import display 
import base64, io
import os

#SECTION - Visualization
#saving the model for future use
# torch.save(learner.Q.state_dict(), "ConvergedModel.pth")
def VisualizeData(agent, env, scores, windowSize, saveFig = True, suppress = False, **kwargs):
	#The parameters used to generate the model will be stored
	parameters = {
		"state_dim":None,
		"action_dim":None,
		"discount":None,
	 	"tau":None,
	 	"lr":None,
	 	"update_freq":None,
	 	"max_size":None,
	 	"batch_size":None,
	 	"gpu_index":None
	}
	for param in parameters:
		if param in kwargs:
			parameters[param] = kwargs[param]
    
	#Creating the output directory, this is a poor implementation but it should work
	root = "./Results"
	listDir = os.listdir(root)
	folderNum = 1
	for dir in listDir:
		if len(dir)<7:
			continue
		if dir[:6] == "Result":
			print(dir, len(dir))
			if folderNum <= int(dir[6]):
				folderNum = int(dir[6])+1
	workDir = root+"/Result"+str(folderNum)
	os.makedirs(workDir)

	#Parameters file:
	paramFile = open(workDir+"/Parameters.txt", "w+")
	for param in parameters:
		paramFile.write(str(param) + ": "+str(parameters[param])+"\n")
	paramFile.close()
	
	#Model
	torch.save(agent.Q.state_dict(), workDir+"/model.pth")
 
	#LearningCurve
	PlotLearningCurve(scores, windowSize, saveFig, suppress, workDir)
	
	#Video
	vidDir = saveVideo(agent, env, workDir)
	if not suppress:
		showVideo(vidDir)
 
def saveVideo(agent, env, saveDir = "."):
	"""Saves a video of the learner operating in the environment

	Args:
		agent (_type_): The Learner
		env (_type_): The Environment
		saveDir (str, optional): Directory in which the video will be saved. Defaults to "./".
	"""
	state, _ = env.reset()
	print("Save Dir", saveDir)
	vid = video_recorder.VideoRecorder(env, path =(saveDir + "/Result.mp4"))
	agent.Q.load_state_dict(torch.load(saveDir + "/model.pth"))
	exit = False
	while not exit:
		env.render()
		vid.capture_frame()
		action = agent.select_action(state)
		state, _, exit, _, _ = env.step(action)
	env.close()
	vid.close()
	return saveDir + "/Result.mp4"

def showVideo(path):
	video = io.open(path, 'r+b').read()
	encoded = base64.b64encode(video)
	display.display(HTML(data='''<video alt="test" autoplay 
                loop controls style="height: 400px;">
                <source src="data:video/mp4;base64,{0}" type="video/mp4" />
             </video>'''.format(encoded.decode('ascii'))))
	#video.close()
	
def PlotLearningCurve(scores, windowSize = 100, saveFig = True, suppress = False, saveDir = "."):
	"""This function is used for displaying the learning rate and saving the results

	Args:
		scores (ndarry): Contains the score per episode
		windowSize (int, optional): Windowsize of the scores. Defaults to 100.
		saveFig (bool, optional): used if you want to save the figure. Defaults to True.
		suppress (bool, optional): Used if you want to suppress showing the figure. Defaults to False.
		saveDir (str, optional): Directory for where to save the figure. Defaults to "./".
	"""
 	#Learning Progress
	fig = plt.figure()
	ax = fig.add_subplot()
	plt.plot(np.arange(len(scores)), scores)
	plt.ylabel("Mean Score of Moving Window of Width: "+ str(windowSize))
	plt.xlabel("Episode Number")
	if not suppress:
		plt.show()
	#Saving it in the Results Directory
	if saveFig:
		plt.savefig(saveDir+"/LearningCurve.png")
 

#SECTION - Deep Q Learning related classes and functions
class ExperienceReplay:
	""" 
	Based on the Replay Buffer implementation of TD3 
	Reference: https://github.com/sfujim/TD3/blob/master/utils.py
	"""
	def __init__(self, state_dim, action_dim,max_size,batch_size,gpu_index=0):
		self.max_size = max_size
		self.ptr = 0
		self.size = 0
		self.state = np.zeros((max_size, state_dim))
		self.action = np.zeros((max_size, action_dim))
		self.next_state = np.zeros((max_size, state_dim))
		self.reward = np.zeros((max_size, 1))
		self.done = np.zeros((max_size, 1))		
		self.batch_size = batch_size
		self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')


	def add(self, state, action,reward,next_state, done):
		self.state[self.ptr] = state
		self.action[self.ptr] = action
		self.next_state[self.ptr] = next_state
		self.reward[self.ptr] = reward
		self.done[self.ptr] = done
		self.ptr = (self.ptr + 1) % self.max_size
		self.size = min(self.size + 1, self.max_size)

	def sample(self):
		ind = np.random.randint(0, self.size, size=self.batch_size)

		return (
			torch.FloatTensor(self.state[ind]).to(self.device),
			torch.FloatTensor(self.action[ind]).long().to(self.device),
			torch.FloatTensor(self.reward[ind]).to(self.device),
			torch.FloatTensor(self.next_state[ind]).to(self.device),
			torch.FloatTensor(self.done[ind]).to(self.device)
		)



class QNetwork(nn.Module):
	"""
	Q Network: designed to take state as input and give out Q values of actions as output
	Documentation Can be Found Here: https://pytorch.org/docs/stable/generated/torch.nn.Module.html
	"""

	def __init__(self, state_dim, action_dim):
		"""
			state_dim (int): state dimension
			action_dim (int): action dimension
		"""
		super(QNetwork, self).__init__()
		self.l1 = nn.Linear(state_dim, 64)
		self.l2 = nn.Linear(64, 64)
		self.l3 = nn.Linear(64, action_dim)
		
	def forward(self, state):
		q = F.relu(self.l1(state))
		q = F.relu(self.l2(q))
		return self.l3(q)



class DQNAgent():

	def __init__(self,
	state_dim, 
	action_dim,
	discount=0.99,
	tau=1e-3,
	lr=5e-4,
	update_freq=4,
	max_size=int(1e5),
	batch_size=64,
	gpu_index=0
	):
		"""
			state_size (int): dimension of each state
			action_size (int): dimension of each action
			discount (float): discount factor
			tau (float): used to update q-target
			lr (float): learning rate
			update_freq (int): update frequency of target network
			max_size (int): experience replay buffer size
			batch_size (int): training batch size
			gpu_index (int): GPU used for training
		"""
		self.state_dim = state_dim
		self.action_dim = action_dim
		self.discount = discount
		self.tau = tau
		self.lr = lr
		self.update_freq = update_freq
		self.batch_size = batch_size
		self.device = torch.device('cuda', index=gpu_index) if torch.cuda.is_available() else torch.device('cpu')


		# Setting up the NNs
		self.Q = QNetwork(state_dim, action_dim).to(self.device)
		self.Q_target = QNetwork(state_dim, action_dim).to(self.device)
		self.optimizer = optim.Adam(self.Q.parameters(), lr=self.lr)

		# Experience Replay Buffer
		self.memory = ExperienceReplay(state_dim,1,max_size,self.batch_size,gpu_index)
		
		self.t_train = 0
	
	def step(self, state, action, reward, next_state, done):
		"""
		1. Adds (s,a,r,s') to the experience replay buffer, and updates the networks
		2. Learns when the experience replay buffer has enough samples
		3. Updates target netowork
		"""
		self.memory.add(state, action, reward, next_state, done)	   
		self.t_train += 1 
					
		if self.memory.size > self.batch_size:
			experiences = self.memory.sample()
			self.learn(experiences, self.discount) #To be implemented
		
		if (self.t_train % self.update_freq) == 0:
			self.target_update(self.Q, self.Q_target, self.tau) #To be implemented 

	def select_action(self, state, epsilon=0.0):
		"""
		TODO: Complete this block to select action using epsilon greedy exploration 
		strategy
		Input: state, epsilon
		Return: Action
		Return Type: int	
		"""
		###### TYPE YOUR CODE HERE ######
		################################# 
		#Currently I dont know if I need to make this a member of the class or not, so Im gonna make state
		#a new var
		state2 = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
		self.Q.eval()
		with torch.no_grad():
			actionVal = self.Q(state2)
		self.Q.train()

		if random.random() > epsilon:
			return np.argmax(actionVal.cpu().data.numpy())
		else:
			return random.choice(np.arange(self.action_dim))

	def learn(self, experiences, discount):
		"""
		TODO: Complete this block to update the Q-Network using the target network
		1. Compute target using  self.Q_target ( target = r + discount * max_b [Q_target(s,b)] )
		2. Compute Q(s,a) using self.Q
		3. Compute MSE loss between step 1 and step 2
		4. Update your network
		Input: experiences consisting of states,actions,rewards,next_states and discount factor
		Return: None
		""" 		
		states, actions, rewards, next_states, dones = experiences
		###### TYPE YOUR CODE HERE ######
		#################################
		#Next maximum value from target network
		QNext = self.Q_target(next_states).detach().max(1)[0].unsqueeze(1)
		#calculating target
		QTarget = rewards + self.discount * QNext*(1-dones)
		#Expected Value
		QExpected = self.Q(states).gather(1, actions)

		#Loss calculation
		loss = F.mse_loss(QExpected, QTarget)
		self.optimizer.zero_grad()
		loss.backward()
		self.optimizer.step()
  
		#Update network
		self.target_update(self.Q, self.Q_target, self.tau)

		                    

	def target_update(self, Q, Q_target, tau):
		"""
		TODO: Update the target network parameters (param_target) using current Q parameters (param_Q)
		Perform the update using tau, this ensures that we do not change the target network drastically
		1. param_target = tau * param_Q + (1 - tau) * param_target
		Input: Q,Q_target,tau
		Return: None
		""" 
		###### TYPE YOUR CODE HERE ######
		#################################
		for targetP, localP in zip(Q_target.parameters(), Q.parameters()):
			targetP.data.copy_(tau*localP.data+(1-tau)*targetP.data)