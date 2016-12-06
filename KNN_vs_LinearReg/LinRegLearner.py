import numpy as np
import sys,csv,math

class LinRegLearner:
    def addEvidence(self, Xtrain, Ytrain):
        Xtrain = np.vstack([Xtrain[:,0], Xtrain[:,1], np.ones(len(Xtrain))]).T
        res = np.linalg.lstsq(Xtrain, Ytrain)
        self.train = res[0]

    def query(self, Xtest):
        Xtest = np.vstack([Xtest[:,0], Xtest[:,1], np.ones(len(Xtest))]).T
        res = np.dot(Xtest,self.train)
        return res

