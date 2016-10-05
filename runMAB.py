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
    maxArms = 10
    
    for ii in range(maxArms):
        addAgent(agentID, bandit, agents, float(maxArms - ii)/maxArms, pow(2,ii), float(ii)/maxArms, armFeatures)
        agentID += 1
    
        
    horizon = 3000
    results = dict()
    results['bandit'] = Result(len(agents), horizon)
    results['random'] = Result(len(agents), horizon)
    results['static'] = Result(len(agents), horizon)
    
    resultBandit = Result(len(agents), horizon)
    resultRandom = Result(len(agents), horizon)
    
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
        productivity = armOutput[0] / armOutput[1]
        bandit.update(productivity) 

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
        armChoice = 4
        #Issue the request to the chosen arm
        armOutput = agents[armChoice].service(lightLevel)
        productivity = armOutput[0] / armOutput[1]
        results['static'].store(ii, armChoice, productivity)
        
        #Remove an arm in the middle of the horizon
        
        if ii == int(horizon/2):
            print 'Removing agent 3'
            removeAgent(3, bandit, agents, armFeatures)
            '''
            print 'Bandit Pulls:', resultBandit.getNbPulls()
            print 'Random Pulls:', resultRandom.getNbPulls()
            print 'Net bandit productivity:', np.sum(resultBandit.rewards)
            print 'Net random productivity:', np.sum(resultRandom.rewards)
            '''
    #Interpret and save results:
    for banditType in results.keys():
        print banditType, ' pulls:', results[banditType].getNbPulls()
        print 'Net ', banditType, ' productivity:', np.sum(results[banditType].rewards)
        appendResults(results[banditType], 'output_%s.csv' % banditType)
'''
    print 'Bandit Pulls:', results['bandit'].getNbPulls()
    print 'Random Pulls:', results['random'].getNbPulls()
    print 'Static Pulls:', results['static'].getNbPulls()

    #for ii in range(len(agents)):
    #    print 'Regret for agent', ii, ':', result.getRegret(agents[ii].pVal)
    print 'Net bandit productivity:', np.sum(results['bandit'].rewards)
    print 'Net random productivity:', np.sum(results['random'].rewards)

    #Append to time history file
    appendResults(results['bandit'], 'output_bandit.csv')
    appendResults(results['random'], 'output_random.csv')
    appendResults(results['static'], 'output_static.csv')
'''
if __name__=='__main__':
    main()
