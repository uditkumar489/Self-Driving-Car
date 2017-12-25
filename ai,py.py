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