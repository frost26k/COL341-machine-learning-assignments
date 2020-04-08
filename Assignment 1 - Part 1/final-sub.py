import sys
import numpy as np
import pandas as pd
from sklearn import linear_model
import math


def part1(train,test,output,weight):
    train = pd.read_csv(train,header=None)
    X = train.iloc[:,0:train.shape[1]-1].values.copy()
    X = np.insert(X,0,1,axis=1)
    Y = train.iloc[:,train.shape[1]-1].values
    
    Xt = X.transpose()
    W1 = np.dot(Xt,X)
    W1 = np.linalg.inv(W1)
    W2 = np.dot(Xt,Y)
     
    W = np.dot(W1,W2)
    
    test = pd.read_csv(test,header=None)
    X_test = test.iloc[:,:].values
    X_test=np.insert(X_test,0,1,axis=1)
    #b = np.hstack((X_test, np.zeros((X_test.shape[0], 1), dtype=X_test.dtype)))
    #b[:,train.shape[1]-1]=1
    predictions = np.dot(X_test,W)
    for p in predictions:
        print(p,file=open(output,"a"))
    for w in W:
        print(w,file=open(weight,"a"))    
    
    
def part2(train,test,paramsinput,output,weight):
    def buildparameters(Xt,X,lamb,Y):
        B1 = np.dot(Xt,X)
        B1=B1 + lamb*np.identity(Xt.shape[0])
        B1 = np.linalg.inv(B1)
        B2 = np.dot(Xt,Y)
        B = np.dot(B1,B2)
        return B
    def mse(X,W,Y):
        pred = np.dot(X,W)
        mse = np.dot((Y-pred).T,(Y-pred))/(sum(Y**2))
        return mse
    def getbestmodel(X,Y,params,Xt,X_test):
        length = X.shape[0]//10
        
        mses = [0]*(len(params))
        copyX = X.copy()
        copyY = Y.copy()
        xtest, xtrain = np.split(copyX, [length])
        ytest, ytrain = np.split(copyY, [length])
        
        for j in range(0,len(params)) :
            W = buildparameters(xtrain.transpose(),xtrain,params[j],ytrain)
            ms = mse(xtest,W,ytest)
            mses[j]=ms
        
        
        for i in range(1,10):
            
            xtest, xtrain[(i-1)*length : (i)*length] = xtrain[(i-1)*length : (i)*length], xtest
            ytest, ytrain[(i-1)*length : (i)*length] = ytrain[(i-1)*length : (i)*length], ytest
                
            for j in range(0,len(params)) :
                W = buildparameters(xtrain.transpose(),xtrain,params[j],ytrain)
                ms = mse(xtest,W,ytest)
                mses[j]=mses[j]+ms
                
     
        min = 0
        for j in range(0,len(params)) :
                mses[j]=mses[j]/10
        
        for i in range(1,len(mses)):
            if mses[i] > mses[min]:
                min=i
        
        W = buildparameters(Xt,X,params[min],Y)
        predictions = np.dot(X_test,W)
        print(params[0])
        print(params[1])
        print(params[2])
        print(mses[0])
        print(mses[1])
        print(mses[2])
        
        print(params[min])
        for p in predictions:
            print(p,file=open(output,"a"))
        for w in W:
            print(w,file=open(weight,"a")) 
    
    
    f = open(paramsinput)
    params=[]
    for ss in f.read().split():
        params.append(float(ss))
    train = pd.read_csv(train,header=None)
    X = train.iloc[:,0:train.shape[1]-1].values.copy()
    X = np.insert(X,0,1,axis=1)
    Y = train.iloc[:,train.shape[1]-1].values
    Xt = X.transpose()
    test = pd.read_csv(test,header=None)
    X_test = test.iloc[:,:].values
    X_test=np.insert(X_test,0,1,axis=1)
    getbestmodel(X,Y,params,Xt,X_test)
    
    
    
def part3(train,test,output):
    def mse(pred,Y):
        mse = np.dot((Y-pred).T,(Y-pred))/(sum(Y**2))
        #mse = np.sum((np.square(pred - Y)))
        #mse=mse/np.sum((np.square(Y)))
        return mse
    def getbestmodel(X,Y,params,Xt,X_test):
        length = X.shape[0]//10
        mses = [0]*(len(params))
        copyX = X.copy()
        copyY = Y.copy()
        xtest, xtrain = np.split(copyX, [length])
        ytest, ytrain = np.split(copyY, [length])
        for j in range(0,len(params)) :
            Lasso = linear_model.LassoLars(alpha=params[j])
            Lasso.fit(xtrain,ytrain)
            pred = Lasso.predict(xtest)
            ms = mse(pred,ytest)
            mses[j]=ms
        for i in range(1,10):
    
            xtest, xtrain[(i-1)*length : (i)*length] = xtrain[(i-1)*length : (i)*length], xtest
            ytest, ytrain[(i-1)*length : (i)*length] = ytrain[(i-1)*length : (i)*length], ytest
                
            for j in range(len(params)) :
                Lasso = linear_model.LassoLars(alpha=params[j])
                Lasso.fit(xtrain,ytrain)
                pred = Lasso.predict(xtest)
                ms = mse(pred,ytest)
                mses[j]=mses[j]+ms
                
        for h in range(len(params)):
            mses[h]=mses[h]/10
        min = 0
        for i in range(len(mses)):
            if mses[i] < mses[min]:
                min=i 
           
        print(params[min])
        Lasso = linear_model.LassoLars(alpha=params[min])
        Lasso.fit(X,Y)
        predictions = Lasso.predict(X_test)
        for p in predictions:
            print(p,file=open(output,"a"))    
    
    params=[0.0000001,0.00001,0.0001,0.003,0.01,0.9,8,50,100,500,1000,2500,4000]
    train = pd.read_csv(train,header=None)
    X = train.iloc[:,0:train.shape[1]-1].values.copy()
    X = np.insert(X,0,1,axis=1)
    #X = np.concatenate((X.copy(),X.copy()**2,1/(1+np.exp(-np.absolute(X.copy())))),axis=1)
    X = np.concatenate((X.copy(),X.copy()**2,np.exp(-np.absolute(X.copy()))),axis=1)
    Y = train.iloc[:,train.shape[1]-1].values
    Xt = X.transpose()
    test = pd.read_csv(test,header=None)
    X_test = test.iloc[:,:].values
    X_test=np.insert(X_test,0,1,axis=1)
    #X_test = np.concatenate((X_test.copy(),X_test.copy()**2,1/(1+np.exp(-np.absolute(X_test.copy())))),axis=1)
    X_test = np.concatenate((X_test.copy(),X_test.copy()**2,np.exp(-np.absolute(X_test.copy()))),axis=1)
    getbestmodel(X,Y,params,Xt,X_test)       
        
    


if __name__ == '__main__':
    if sys.argv[1] == 'a':
        part1(*sys.argv[2:])
    if sys.argv[1] == 'b':
        part2(*sys.argv[2:]) 
    if sys.argv[1] == 'c':
        part3(*sys.argv[2:])     
