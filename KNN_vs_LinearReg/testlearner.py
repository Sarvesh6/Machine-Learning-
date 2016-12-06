import numpy as np
import sys, csv, math
import matplotlib.pyplot as plt
import KNNLearner
import LinRegLearner

infile = open("apple2.csv", 'rU')
has_header = csv.Sniffer().has_header(infile.read(1024))
infile.seek(1)  # rewind
reader = (csv.reader(infile,delimiter=','))
if has_header:
   next(reader)
data = list(reader)
rows = len(data)
cols = len(data[1])-2
xyz=np.floor(0.6*rows)
pqr=rows-xyz
Xtrain = np.zeros(((xyz),cols))
Ytrain = np.zeros(xyz)
Xtest = np.zeros(((pqr),cols))
Ytest = np.zeros(pqr)

test_cnt = 0
train_cnt = 0
cnt = 0
X = np.zeros((rows,cols))
Y = np.zeros(rows)
for row in data:
    if cnt < np.floor(0.6*rows):
        i=0
        while i!=cols:
            Xtrain[train_cnt,i] = row[i+1]
            i=i+1
        Ytrain[train_cnt] = row[i+1]
        train_cnt += 1
    else:
        i=0
        while i!=cols:
            Xtest[test_cnt,i] = row[i+1]
            i=i+1
        Ytest[test_cnt] = row[i+1]
        test_cnt += 1
    i=0
    while i!=cols:
        X[cnt,i] = row[i+1]
        i=i+1
    Y[cnt] = row[i+1]  
    cnt +=1
    
##print Xtrain[0,:],Ytrain[0]
#print "XTRAIN:",(np.array(Xtrain))
#print "YTRAIN:",Ytrain
#print "XTEST:",np.matrix(Xtest)
#print "YTEST:",Ytest
#print len(X)
#print len(Y)
#print "X:",np.matrix(X)
#print "Y:",Y
n =rows-1
rms_knn_in = np.zeros((n))
rms_knn_out = np.zeros((n))
corr_coef_knn_in = np.zeros((n))
corr_coef_knn_out = np.zeros((n))

rms_lr_in = 0
corr_coef_lr_in = 0
rms_lr_out = 0
corr_coef_lr_out = 0
p=0
plt.clf()
plt.plot(Xtrain)
plt.show()
K = np.zeros((n))
Y_best_knn = []
for k in range(1,n+1):
    K[k-1] = k
    learner = KNNLearner.KNNLearner(k)
    learner.addEvidence(Xtrain,Ytrain)
    Y_out_knn = learner.query(Xtest)
    
    sum = 0
    for i in range(len(Y_out_knn)):
        sum += math.pow((Y_out_knn[i] - Ytest[i]),2)
    rms_knn_out[k - 1] = math.sqrt(sum/len(Y_out_knn))
    corr_coef_knn_out[k - 1] = np.corrcoef(Y_out_knn,Ytest)[0,1]

    learner.addEvidence(X,Y)
    Y_in_knn = learner.query(Xtest)
    
    sum = 0
    for i in range(len(Y_in_knn)):
        sum += math.pow((Y_in_knn[i] - Ytest[i]),2)
    rms_knn_in[k - 1] = math.sqrt(sum/len(Y_in_knn))
    corr_coef_knn_in[k - 1] = np.corrcoef(Y_in_knn,Ytest)[0,1]
    ##print k," ",rms_knn_in[k - 1]
    ##print k,"-",rms_knn_out[k - 1],"-",corr_coef_knn_out[k - 1],"-",rms_knn_in[k - 1],"-",corr_coef_knn_in[k - 1]
    if k == 2 or (rms_knn_in[k-1] < rms_knn_in[p-1] ) or (rms_knn_in[k-1] >= rms_knn_in[p-1] and corr_coef_knn_in[k-1]>corr_coef_knn_in[p-1]):
        Y_best_knn = Y_out_knn
        p=k

learner = LinRegLearner.LinRegLearner()
learner.addEvidence(Xtrain, Ytrain)
Y_out_lr = learner.query(Xtest)

sum = 0
for i in range(len(Y_out_lr)):
    sum += math.pow((Y_out_lr[i] - Ytest[i]),2)
rms_lr_out = math.sqrt(sum/len(Y_out_lr))
corr_coef_lr_out = np.corrcoef(Y_out_lr,Ytest)[0,1]

learner.addEvidence(X, Y)
Y_in_lr = learner.query(Xtest)

sum = 0
for i in range(len(Y_in_lr)):
    sum += math.pow((Y_in_lr[i] - Ytest[i]),2)
rms_lr_in = math.sqrt(sum/len(Y_in_lr))
corr_coef_lr_in = np.corrcoef(Y_in_lr,Ytest)[0,1]

print "In Sample - RMS Error for Linear Regression - " + str(rms_lr_in)
print "In Sample - Correlation Coefficient for Linear Regression - " + str(corr_coef_lr_in)
print "Out Sample - RMS Error for Linear Regression - " + str(rms_lr_out)
print "Out Sample - Correlation Coefficient for Linear Regression - " + str(corr_coef_lr_out)
print ""
print ""
print "Best value for k in knn is ",p

plt.clf()
plt.scatter(Ytest ,Y_out_lr)
plt.legend(['Lin_Reg Y'])
plt.ylabel("Predicted Y")
plt.xlabel("Actual Y")
plt.savefig('predicted_vs_actual_lr.pdf', format='pdf')
plt.show()

plt.clf()
plt.plot(K,rms_knn_out, K, rms_knn_in)
plt.legend(['RMSE KNN Out', 'RMSE KNN In'])
plt.ylabel("Root Mean Square Error")
plt.xlabel("K")
plt.savefig('rmse_in_out.pdf', format='pdf')
plt.show()

plt.clf()
plt.scatter(Ytest, Y_best_knn)
plt.legend(['KNN Y'])
plt.ylabel("Predicted Y")
plt.xlabel("Actual Y")
plt.savefig('predicted_vs_actual_knn.pdf', format='pdf')
plt.show()

plt.clf()
plt.scatter(Y_best_knn ,Y_out_lr)
plt.legend(['Predicted Y'])
plt.ylabel("Linear Regression Y")
plt.xlabel("Knn Y ")
plt.savefig('predicted_knn_vs_prdicted_lr.pdf', format='pdf')
plt.show()

