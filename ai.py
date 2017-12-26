# -*- coding: utf-8 -*-
"""
Created on Mon Dec 25 14:22:09 2017

@author: Udit
"""

# AI for Self Driving Car

# Importing the libraries

import numpy as np                   #for dealing with arrays
import random                        #for creating some randomization in the env
import os                            #for saving the brain state of the agent
import torch                         #for implementing neuran-networks also it is capable of handling dynamic graphs
import torch.nn as nn                #contains essential tools to implement neural-network
import torch.nn.functional as F      #contains imp funcs such as LOSS FUNCTION 
import torch.optim as optim          #for implementing OPTIMIZERS
import torch.autograd as autograd    
from torch.autograd import Variable  #to convert tensors into variables and gradients


#AIM_1 -> To create neural network 
 #Our Neural-Network class will have 2 functions :
 #1. init func()    - to define the structure of the NN
 #2. forward func() - to activate the neurons and return q_values for each possible action
 
class Network(nn.Module):
    
    def __init__(self, input_size, nb_action):  #'python' syntax to define class variables
        super(Network, self).__init__()         #trick to inherit all the properties of "nn.Module" in single shot
        self.input_size = input_size            #self.input_size is just a way to attact a variable to the input layer of neuralN. 
        #Note : above line signifies no. of input neuron = input size (in this case i.e 5)
        self.nb_action = nb_action              #same as above case but in o/p neurons
        self.fc1 = nn.Linear(input_size, 30)    #full connection b/w i/p and hidden layer
        self.fc2 = nn.Linear(30, nb_action)     #full connection b/w hidden and o/p layer
        
        
    def forward(self, state):        #ofcourse , states are the inputs of our NN
        x = F.relu(self.fc1(state))  #activating hidden_layer neurons
        q_values = self.fc2(x)       #activating o/p neurons
        return q_values
    


#AIM_2 -> To Implement Experience Replay i.e. to store certain transitions / events for experience
  #Exp_Rep class will have 3 funcs() :
  #1. init func()      - to declare class variables 
  #2. push func()      - to append memory + make sure memory never exceeed 'capacity'  
  #3. sample func()    - to take random samples of events from the memory     

class ReplayMemory(object):
    
    def __init__(self, capacity):               #capacity = num of events to be stored
        self.capacity = capacity
        self.memory = []
        
    def push(self, event):                      #event will have a format of a touple wid 4 eles - last & next state , last action , last reward 
        self.memory.append(event)               #to apped the memory with events
        if len(self.memory) > self.capacity:    #deleting the 1st event from memoery whenever new event attempts in memory
            del self.memory[0]
            
    def sample(self, batch_size):               #batch_size = size of sample (explanation in README.md)
        samples = zip(*random.sample(self.memory, batch_size))
        return map(lambda x: Variable(torch.cat(x, 0)), samples)
    
    
    
#AIM_3 -> To Implement Deep Q Learning
   #Deep_q_Net class will have 5 functions() :
   #1. init func()       - to declare variables and define class structure

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):                      #to apply stochastic gradient descent
        self.gamma = gamma
        self.reward_window = []
        self.model = Network(input_size, nb_action)
        self.memory = ReplayMemory(100000)                                 #memory self-assumed
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)   #creartin optimizer (ADAM in this case)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0                                               #action can have values 0,1,2 corrsp rotation
        self.last_reward = 0                                               #reward can have values 0,+1,-1