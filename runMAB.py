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
from math import *
import csv

def addAgent(id, bandit, agents, minLight, iota, pVal, features):
    print 'Adding agent with minlight:', minLight, ' iota:', iota, ' and p:', pVal
    agents[id] = Agent(minLight, iota, pVal)
    bandit.addArm(id, 1)
    features[id] = iota
    
def removeAgent(id, bandit, agents, features):
    bandit.removeArm(id)
    del agents[id]
    del features[id]

def appendResults(result, fileName):
    outRow = ",".join(str(ii) for ii in result.getTimeHistoryRow())
    banditFile = open(fileName,'a')
    banditFile.write(outRow + '\n')
    banditFile.close()

def main():
    agents = dict()
    armFeatures = dict()
    delta = 0.05 #Eq 4 from Li2010
    alpha = 1 + sqrt(log(2/delta)/2)

    print 'Using alpha of ', alpha
    bandit = HybridUCB(alpha, 1)

    
    agentID = 0
    maxArms = 5
    
    for ii in range(maxArms):
        addAgent(agentID, bandit, agents, float(maxArms - ii)/maxArms, pow(4,ii), float(ii+1)/maxArms, armFeatures)
        agentID += 1
    
        
    horizon = 2000
    results = dict()
    results['bandit'] = Result(len(agents), horizon)
    results['random'] = Result(len(agents), horizon)
    results['static'] = Result(len(agents), horizon)
    #results['static2'] = Result(len(agents), horizon)

    staticChoice = 1
    for ii in range(horizon):
        #Determine the light level (ranges from 0 to 1):
        #This is the environmental feature
        lightLevel = np.random.rand(1)

        #armFeatures has the arm-specific feature vectors (in this case, the iota values)
        armChoice = bandit.select(lightLevel, armFeatures)

        #print 'Pulling: ', armChoice

        #Issue the request to the chosen arm
        armOutput = agents[armChoice].service(lightLevel)

        #Update the bandit with the outcome
        productivity = armOutput[0] #/ armOutput[1]
        bandit.update(productivity) 
        productivity = armOutput[0] / armOutput[1]
        #Store result
        results['bandit'].store(ii, armChoice, productivity)

        #Run a random selection
        randIndex = int(np.random.rand(1)*len(agents))
        armChoice = agents.keys()[randIndex] #since we might remove something in the middle
        #print 'Random choice:', armChoice
        
        #Issue the request to the chosen arm
        armOutput = agents[armChoice].service(lightLevel)
        productivity = armOutput[0] / armOutput[1]
        results['random'].store(ii, armChoice, productivity)


        #Run a static selection of a sub-optimal choice
        armChoice = staticChoice
        #Issue the request to the chosen arm
        armOutput = agents[armChoice].service(lightLevel)
        productivity = armOutput[0] / armOutput[1]
        results['static'].store(ii, armChoice, productivity)

        '''
        #Run a static selection of a sub-optimal choice
        armChoice = 2
        #Issue the request to the chosen arm
        armOutput = agents[armChoice].service(lightLevel)
        productivity = armOutput[0] / armOutput[1]
        results['static2'].store(ii, armChoice, productivity)
        '''

        #Alter arm composition in the middle of the test
        #Remove an arm in the middle of the horizon
        
        if ii == int(horizon/2):
            print 'Removing agent ', staticChoice
            removeAgent(staticChoice, bandit, agents, armFeatures)
            print 'Changing static choice to adjust'
            staticChoice = 4
            for banditType in results.keys():
                print banditType, ' pulls:', results[banditType].getNbPulls()

            
    #Interpret and save results:
    print 'Final results:'
    for banditType in results.keys():
        print banditType, ' pulls:', results[banditType].getNbPulls()
        print 'Net ', banditType, ' productivity:', np.sum(results[banditType].rewards)
        appendResults(results[banditType], 'output_%s.csv' % banditType)

if __name__=='__main__':
    main()
