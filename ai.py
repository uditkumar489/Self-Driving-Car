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
        

    def forward(self, state):                   #ofcourse , states are the inputs of our NN
        x = F.relu(self.fc1(state))             #activating hidden_layer neurons
        q_values = self.fc2(x)                  #activating o/p neurons
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
   #Deep_q_Net class will have 7 functions() :
   #1. init()           - to declare variables and define class structure
   #2. select_action()  - to selct which action to perform next
   #3. learn()          - to backpropagate errors and update the weights
   #4. update()         - to update everything as the AI reaches new state + return new action
   #5. score()          - to calculate the mean of award window
   #6. save()           - to save the state of brain
   #7. load()           - to load the last saved brain

class Dqn():
    
    def __init__(self, input_size, nb_action, gamma):                       #to apply stochastic gradient descent
        self.gamma = gamma
        self.reward_window = []                                             #contains list of rewads for a time-interval
        self.model = Network(input_size, nb_action)                         #making object of neural class
        self.memory = ReplayMemory(100000)                                  #memory self-assumed
        self.optimizer = optim.Adam(self.model.parameters(), lr = 0.001)    #creartin optimizer (ADAM in this case)
        self.last_state = torch.Tensor(input_size).unsqueeze(0)
        self.last_action = 0                                                #action can have values 0,1,2 corrsp rotation
        self.last_reward = 0                                                #reward can have values 0,+1,-1
        
    def select_action(self, state):
        probs = F.softmax(self.model(Variable(state, volatile = True))*100) #Temp-Parameter=100
        action = probs.multinomial()                                        #to have a rondom_draw over the above probability distribution
        return action.data[0,0]
    
    def learn(self, batch_state, batch_next_state, batch_reward, batch_action):
        outputs = self.model(batch_state).gather(1, batch_action.unsqueeze(1)).squeeze(1)
        next_outputs = self.model(batch_next_state).detach().max(1)[0]
        target = self.gamma*next_outputs + batch_reward
        td_loss = F.smooth_l1_loss(outputs, target)                         #HUBER_LOSS (i.e smooth_ll_loss) recommended in deepL
        self.optimizer.zero_grad()                                          #this will re-initialize the optimizer
        td_loss.backward(retain_variables = True)                           #this will backpropagate the loss
        self.optimizer.step()                                               #this will update the weights
        
    def update(self, reward, new_signal):                                   #to update last & new state , last action and reward
        new_state = torch.Tensor(new_signal).float().unsqueeze(0)
        self.memory.push((self.last_state, new_state, torch.LongTensor([int(self.last_action)]), torch.Tensor([self.last_reward])))
        action = self.select_action(new_state)                              #performing new action
        if len(self.memory.memory) > 100:
            batch_state, batch_next_state, batch_action, batch_reward = self.memory.sample(100)
            self.learn(batch_state, batch_next_state, batch_reward, batch_action)
        self.last_action = action                                           #updating last action
        self.last_state = new_state                                         #updating last state
        self.last_reward = reward                                           ##updating last reward
        self.reward_window.append(reward)
        if len(self.reward_window) > 1000:
            del self.reward_window[0]
        return action                                                       #returning the new action performed
        
    def score(self):
        return sum(self.reward_window)/(len(self.reward_window)+1)          #+1 to avoid 'dividing by 0' 
    
    def save(self):                                                         #to save the brain we need to save the state and weights only
        torch.save({'state_dict': self.model.state_dict(),                  #here we are using Python-dictionary saving technique
                    'optimizer' : self.optimizer.state_dict(),              #since weights are associated with optimizer
                   }, 'saved_brain.pth')                                    #file name in which the last brain will be stored 
    
    def load(self):
        if os.path.isfile('saved_brain.pth'):                               #checking if the last brain file exists
            print("=> loading last brain state... ")
            lastBrain = torch.load('saved_brain.pth')                       #variabale to load the filedata
            self.model.load_state_dict(lastBrain['state_dict'])             #loading corresponding values to the keys (here to 'state_dict')
            self.optimizer.load_state_dict(lastBrain['optimizer'])
            print("And it's done !")
        else:
            print("no last brain found...")