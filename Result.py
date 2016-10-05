# -*- coding: utf-8 -*-
'''Utility class for handling the results of a Multi-armed Bandits experiment.'''

__author__ = "Olivier Cappé, Aurélien Garivier"
__version__ = "$Revision: 1.7 $"


import numpy as np

class Result:
    """The Result class for analyzing the output of bandit experiments."""
    def __init__(self, nbArms, horizon=1000):
        self.nbArms = nbArms
        self.choices = np.zeros(horizon)
        self.rewards = np.zeros(horizon)
        self.maxT = 0
    def store(self, t, choice, reward):
        if t >= self.choices.shape[0]:
            np.append(self.choices, choice)
            np.append(self.rewards, reward)
        else:
            self.choices[t] = choice
            self.rewards[t] = reward
            self.maxT = t
    def getNbPulls(self):
        if (self.nbArms==float('inf')):
            self.nbPulls=np.array([])
            pass
        else :

            #This needs to be adjusted to support the case when an arm is removed in the middle of a run -
            #Ahh - when asking for interim results, the zero default is reckoned with arm 0 as well...
            nbPulls = np.zeros(self.nbArms)
            #nbPulls = dict()
            #print 'Choices:', self.choices
            for t in range(self.maxT):
                nbPulls[self.choices[t]] += 1
            return nbPulls
    
    def getRegret(self, bestExpectation):
        return np.cumsum(bestExpectation-self.rewards)

    def getTimeHistoryRow(self):
        #Restructure the .rewards array as a list row, indexed by timestep
        #Output for Matlab scripts to analyze...
        return np.cumsum(self.rewards).tolist()
