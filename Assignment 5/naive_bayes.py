import sys
import pandas as pd
import numpy as np
import math
import re
from collections import Counter


def mainP(trainin,testin,pred):
    
    
    traindata = pd.read_csv(trainin)
    
    positiveC = Counter()
    negativeC = Counter()
    
    for i in range(0,traindata.shape[0]):
        traindata['review'][i] = traindata['review'][i].lower()
        traindata['review'][i] = re.split('\W+', traindata['review'][i])
        traindata['review'][i] = Counter(traindata['review'][i])
        if traindata['sentiment'][i] == 'positive':
            positiveC = positiveC + traindata['review'][i]
        if traindata['sentiment'][i] == 'negative':
            negativeC = negativeC + traindata['review'][i]
    
    X = traindata.values.copy()
    Xtrain = X[:,0].copy()
    
    Ytrain = X[:,1].copy()
    Ytrain[Ytrain == 'positive'] = 1
    Ytrain[Ytrain == 'negative'] = 0
            
    ptotal = sum(positiveC.values())
    ntotal = sum(negativeC.values())        
    positiveR = len(Ytrain[Ytrain==1])/Xtrain.shape[0]
    negativeR = len(Ytrain[Ytrain==0])/Xtrain.shape[0]
    
    testdata = pd.read_csv(testin)
    for i in range(0,testdata.shape[0]):
        testdata['review'][i] = testdata['review'][i].lower()
        testdata['review'][i] = re.split('\W+', testdata['review'][i])
        testdata['review'][i] = Counter(testdata['review'][i])
    XF = testdata.values.copy()
    Xtest = XF[:,0].copy()
    
    predictions = np.zeros((Xtest.shape[0],1))
    t1 = len(positiveC.most_common())
    t2 = len(negativeC.most_common())
    
    for i in range(0,Xtest.shape[0]):
        pos, neg = 0, 0
        iterator = Xtest[i].most_common()
        for w in iterator:
            pos = pos + np.log(1 + positiveC[w[0]]) + np.log(positiveR) - np.log(ptotal + t1)
            neg = neg + np.log(1 + negativeC[w[0]]) + np.log(negativeR) - np.log(ntotal + t2)
            
        if pos > neg : 
            predictions[i]=1
        else : 
            predictions[i]=0
            
            
    for pr in predictions:
        print(np.asscalar(pr),file=open(pred, "a"))        

if __name__ == '__main__':
    mainP(*sys.argv[1:])  