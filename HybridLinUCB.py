#!/usr/bin/env python2.7

#Hybrid LinUCB implementation, following example of https://github.com/Fengrui/HybridLinUCB-python.git, but with (sequential) code instead of vectorized for ease of reading
#Inmplementation of Li2010Contextual, A Contextual-Bandit Approach to Personalized News Article Recommendation, Algorithm 2

import numpy as np
import math
import sys
from scipy import linalg

class HybridArm:
    #Representation of each arm. Has a set of arm-specific features of size d, common features k
    def __init__(self, id, d, k, alpha):
        self.id = id
        self.d = d
        self.k = k
        self.alpha = alpha
        #Li lines 8-10
        self.A = np.identity(self.d)
        self.B = np.zeros((self.d, self.k))
        self.b = np.zeros((self.d, 1))

    def getA(self):
        return self.A
    def getB(self):
        return self.B
    def getb(self):
        return self.b
    def getID(self):
        return self.id
    
    def getP(self, A0, b0, betaHat, z, x):
        #Li Lines 12-14, args are numpy arrays
        self.Ainv = linalg.inv(self.A)
        self.A0inv = linalg.inv(A0)
        self.thetaHat = np.dot(self.Ainv, self.b - np.dot(self.B, betaHat))
        self.x = x
        self.z = z
        
        #I have no doubt there is a better way to write these matrix products
        self.s1 = np.dot(np.transpose(z), np.dot(self.A0inv, z))
        self.s2 = np.dot(np.transpose(z), np.dot(self.A0inv, np.dot(np.transpose(self.B), np.dot(self.Ainv, x))))
        self.s3 = np.dot(np.transpose(x), np.dot(self.Ainv, x))
        self.s4 = np.dot(np.transpose(x), np.dot(self.Ainv, np.dot(self.B, np.dot(self.A0inv, np.dot(np.transpose(self.B), np.dot(self.Ainv, x))))))

        self.s = self.s1 - 2*self.s2 + self.s3 + self.s4 #Li line 13

        self.p = np.dot(np.transpose(z), betaHat) + np.dot(np.transpose(x), self.thetaHat) + self.alpha*np.sqrt(self.s)
        return self.p

    def update(self, reward):
        self.A += np.dot(self.x, np.transpose(self.x))
        self.B += np.dot(self.x, np.transpose(self.z))
        self.b += reward * self.x
        
class HybridUCB:
    def __init__(self, ucb, env_feats):
        self.alpha = ucb; #upper bound coefficient, line 14 in the Li paper
        self.k = env_feats; #size of environment features common to all arms
        self.A0 = np.identity(self.k); #initialization of env features, line 1
        self.b0 = np.zeros((self.k,1)); #line 2


        self.z = np.zeros((self.k, 1))

        #Maintain the set of arms in the system currently
        self.arms = dict()
        self.currentArm = None
        
    #Todo: bulk add / remove 
    def addArm(self, id, fLen):
        #Add an arm to the system with unique ID id, with feature length fLen:
        self.arms[id] = HybridArm(id, fLen, self.k, self.alpha)

    def removeArm(self, id):
        try:
            del self.arms[id]
        except KeyError:
            print 'Attempted to remove nonexisted arm id', id
            
        
    def select(self, z, x):
        #Call the arm-specific code
        #x is a dict of vectors of arm-specific features, z is the vector of common features
        #arms is a list of current arms of type HybridArm to choose from
        
        #Save the z for later:
        self.z = z


        self.betaHat = np.dot(linalg.inv(self.A0), self.b0)
        bestP = dict()
        for theArm in self.arms:
            bestP[theArm] = self.arms[theArm].getP(self.A0, self.b0, self.betaHat, z, x[theArm])

        #Choose arm with max P, ties broken arbitrarily
        #http://stackoverflow.com/questions/268272/getting-key-with-maximum-value-in-dictionary
        self.currentArm = self.arms[max(bestP, key=lambda k: bestP[k])]
        return self.currentArm.getID()
        
    def update(self, reward):
        #The class is stateful - expects the self.currentArm to have been pulled and produced a real-valued reward
        #lines 17-18
        self.A0 += np.dot(np.transpose(self.currentArm.getB()),  np.dot(linalg.inv(self.currentArm.getA()), self.currentArm.getB()))
        self.b0 += np.dot(np.transpose(self.currentArm.getB()), np.dot(linalg.inv(self.currentArm.getA()), self.currentArm.getb()))

        #Update the arm-specific matrices: lines 19-21
        self.currentArm.update(reward)

        #Update the general matrices again: lines 22-23
        self.A0 += np.dot(self.z, np.transpose(self.z))
        self.A0 -= np.dot(np.transpose(self.currentArm.getB()), np.dot(linalg.inv(self.currentArm.getA()), self.currentArm.getB()))

        self.b0 += reward * self.z
        self.b0 -= np.dot(np.transpose(self.currentArm.getB()), np.dot(linalg.inv(self.currentArm.getA()), self.currentArm.getb()))
        
        
        
