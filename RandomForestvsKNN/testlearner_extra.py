import numpy as np
import sys, csv, math, random
import matplotlib.pyplot as plt
import KNNLearner
import RandomForestLearner
import RandomForestLearnerBoost

infile = open("apple.csv", 'rU')
has_header = csv.Sniffer().has_header(infile.read(1024))
infile.seek(1)  # rewind
reader = (csv.reader(infile,delimiter=','))
if has_header:
   next(reader)
data = list(reader)
rows = len(data)
cols = len(data[1])-1
xyz=np.floor(0.6*rows)
pqr=rows-xyz
Xtrain = np.zeros((xyz,cols))
Ytrain = np.zeros((xyz,1))
Xtest = np.zeros((pqr,cols))
Ytest = np.zeros((pqr,1))

test_cnt = 0
train_cnt = 0
cnt = 0
X = np.zeros((rows,cols))
Y = np.zeros((rows,1))
for row in data:
    if cnt < np.floor(0.6*rows):
        i=1
        while i!=cols+1:
            Xtrain[train_cnt,i-1] = row[i]
            i=i+1
        i=i-1
        Ytrain[train_cnt] = row[i]
        train_cnt += 1
    else:
        i=1
        while i!=cols+1:
            Xtest[test_cnt,i-1] = row[i]
            i=i+1
        i=i-1
        Ytest[test_cnt] = row[i]
        test_cnt += 1
    i=1
    while i!=cols+1:
        X[cnt,i-1] = row[i]
        i=i+1
    i=i-1
    Y[cnt] = row[i]  
    cnt +=1
n = 11
rms_rf1_in = np.zeros((n))
rms_rf1_out = np.zeros((n))

rms_rf_in = np.zeros((n))
rms_rf_out = np.zeros((n))

K = np.zeros((n))
corr_rf = np.zeros((n))
corr_rf1 = np.zeros((n))
for k in range(90,101):

    K[k-90] = k

    #out of sample random forest
    learner = RandomForestLearner.RandomForestLearner(k)
    d = np.hstack([Xtrain,Ytrain])
    learner.addEvidence(d[:0.6*len(d),:])
    Y_out_rf = learner.query(Xtest)
    corr_rf[k-90] = np.corrcoef(Y_out_rf,Xtest[:,-1])[0,1]

    #out of sample random forest1
    learner = RandomForestLearnerBoost.RandomForestLearner(k)
    d = np.hstack([Xtrain,Ytrain])
    learner.addEvidence(d[:0.6*len(d),:])
    Y_out_rf1 = learner.query(Xtest)
    corr_rf1[k-90] = np.corrcoef(Y_out_rf1,Xtest[:,-1])[0,1]

plt.clf()
print len(Y_out_rf1), len(Y_out_rf)
plt.plot(K, corr_rf, K, corr_rf1)
plt.legend(['CORCOEFF RF', 'CORCOEFF RF Improved'])
plt.ylabel("CORCOEFF")
plt.xlabel("K")
plt.savefig('corcoeff_rf_rf1.pdf', format='pdf')
