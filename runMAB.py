#/usr/bin/env python2.7

#Test rig for the contextual MAB experiment
#We have a series of agents that can provide assistance, and a HybridLinUCB
#that learns from the repeated draws

#Environment (shared feature): Light level, continuous real
#Per-arm feature: iota value, corresponding to distance to assistance

#Setup: Arms have internal parameters set such that a greater iota incurs a greater cost, but has a greater chance of success in execution

from Agent import Agent
from HybridLinUCB import HybridUCB
import numpy as np
from Result import *

def addAgent(id, bandit, agents, minLight, iota, pVal, features):
    agents[id] = Agent(minLight, iota, pVal)
    bandit.addArm(id, 1)
    features[id] = iota
    
def main():
    agents = dict()
    armFeatures = dict()
    bandit = HybridUCB(0.3, 1)

    agentID = 0
    addAgent(agentID, bandit, agents, 1, 4, 0.2, armFeatures)
    agentID += 1
    addAgent(agentID, bandit, agents, 1, 8, 0.6, armFeatures)
    agentID += 1
    addAgent(agentID, bandit, agents, 1, 16, 0.9, armFeatures)

    horizon = 1000
    result = Result(len(agents), horizon)
    
    for ii in range(horizon):
        #Determine the light level (ranges from 0 to 10):
        #This is the environmental feature
        lightLevel = np.random.rand(1) * 10

        #armFeatures has the arm-specific feature vectors (in this case, the iota values)
        armChoice = bandit.select(lightLevel, armFeatures)

        #print 'Pulling: ', armChoice

        #Issue the request to the chosen arm
        armOutput = agents[armChoice].service(lightLevel)

        #Update the bandit with the outcome
        productivity = armOutput[0] / armOutput[1]
        bandit.update(productivity) 

        #Store result
        result.store(ii, armChoice, productivity)

    #Interpret Results:
    print 'Pulls:', result.getNbPulls()

    #for ii in range(len(agents)):
    #    print 'Regret for agent', ii, ':', result.getRegret(agents[ii].pVal)
    print 'Net productivity:', np.cumsum(result.rewards)[len(result.rewards)-1]
if __name__=='__main__':
    main()
