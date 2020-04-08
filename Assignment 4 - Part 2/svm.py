import sys
import pandas as pd
import numpy as np
import timeit
import math

def svm(Xtrain,i,j,size):
    
    class1 = Xtrain[np.where(Xtrain[:,-1] == i)]
    class2 = Xtrain[np.where(Xtrain[:,-1] == j)]
    class1[:,-1] = 1
    class2[:,-1] = -1
    miniSo = np.concatenate((class1,class2), axis=0)
    np.random.shuffle(miniSo)
    W = np.zeros((784,1))
    b = 0
    for p in range(0,500):
        for q in range(0,int(size/150)):
            Xb = miniSo[q*150:(q+1)*150, :-1]
            Yb = miniSo[q*150:(q+1)*150, -1]
            print(Yb.shape)
            if (Yb.shape[0] == 150):
                con = (Xb@W + b)*(Yb.reshape(150,1))
                con2 = Yb.reshape(150,1)*(con<=1)
                con3 = 1/(0.01*(p+1))
                W = (1 - (0.01*con3))*W + (np.sum((con3/150)*con2*Xb,axis=0).reshape(784,1))
                b = b + np.sum((con3/150)*con2)
    return W,b        
            
def mainP(trainin,testin,pred):
    
    traindata = pd.read_csv(trainin,header=None)
    X = traindata.values.copy()
    Xtrain = X[:,0:X.shape[1]-1].copy()/255.0
    Ytrain = X[:,X.shape[1]-1].copy()
    Xtrain = np.concatenate((Xtrain,Ytrain.reshape(Xtrain.shape[0],1)),axis=1)
    Xtrain.shape
    
    testdata = pd.read_csv(testin,header=None)
    XF = testdata.values.copy()/255.0
    Xtest = XF[:,0:XF.shape[1]-1].copy()
    predictions = np.zeros((Xtest.shape[0],10))
    
    for i in range(0,10):
        for j in range(i+1,10):
            
            W, b = svm(Xtrain,i,j,Xtest.shape[0])
            dum = Xtest@W + b
            dum = (2*((dum)>0))-1
            for u in range(0,testdata.shape[0]):
                if dum[u] == 1:
                    predictions[u,i] = predictions[u,i] + 1
                else :
                    predictions[u,j] = predictions[u,j] + 1 
    final = np.argmax(predictions,axis=1)   
    for pr in final:
        print(np.asscalar(pr),file=open(pred, "a"))
        
if __name__ == '__main__':
    mainP(*sys.argv[1:])        
