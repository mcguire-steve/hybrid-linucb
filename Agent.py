#/usr/bin/env python2.7

#Sample agent for testing the LinUCB code
from random import random
import numpy as np

class Agent:
    def __init__(self, minLight, iota, pVal):
        self.minLight = minLight
        self.iota = iota
        self.pVal = pVal


    def service(self, lightLevel):
        #Service a request for help, returning a tuple of 
        #(reward, cost) for the assistance

        reward = 0
        if lightLevel > self.minLight:
            outcome = float(random() < self.pVal) #Bernoulli model
            reward = outcome  #add continuous rewards...

        cost = self.iota #+ np.random.rand(1)
        return (reward, cost)
    
            
