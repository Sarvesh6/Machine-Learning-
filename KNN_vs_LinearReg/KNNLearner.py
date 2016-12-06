import numpy as np
import sys, csv, math
class KNNLearner:
    def __init__(self,k):
        self.k = k

    def addEvidence(self, Xtrain, Ytrain):
        self.Xtrain = Xtrain
        self.Ytrain = Ytrain
        

    def query(self,Xtest):
        n_test = len(Xtest)
        n_train = len(self.Xtrain)
        Y = np.zeros(n_test)
        for i in range(n_test):
            dist = []
            for j in range(n_train):
                d = 0
                for k in range(len(Xtest[i])):
                    diff = Xtest[i][k] - self.Xtrain[j][k]
                    d += math.pow(diff,2)
                dist.append((math.sqrt(d),j))
            dist.sort()
            neighbours = []
            for k in range(self.k):
                try:
                    neighbours.append(self.Ytrain[dist[k][1]])
                except IndexError:
                    pass
            Y[i] = np.mean(neighbours)
        return Y
