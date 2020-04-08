import numpy as np
import pandas  as pd
import sys
import math
from numpy import linalg as LA

def part1(trainin,testin,paramsinput,output,weight):
    def entropy(weights,Xtrain,Ytrain):
        p = np.sum(Ytrain*np.log(predict(Xtrain,weights)))
        p /= 2*Xtrain.shape[0]
        return p
       
    
    def predict(X,weigh):
        temp=np.dot(X,weigh)
        temp=np.exp(temp)
        s=(np.sum(temp,axis=1)).reshape(temp.shape[0],1)
        return temp/s
        
    def error(X,Y,Ypred):
        return (X.T)@(Y-Ypred)
    train = pd.read_csv(trainin,header=None)
    test = pd.read_csv(testin,header=None)
    # train.drop(train.columns[[8]], axis=1, inplace=True)
    
    
    X = train.iloc[:,:].values.copy()
    XX = test.iloc[:,:].values.copy()
    for i in range(0,test.shape[0]):
        if XX[i,5]=='convenient':
            XX[i,5]=='cconvenient'
        if XX[i,4]=='critical':                   
            XX[i,4]=='ccritical' 
    for i in range(0,train.shape[0]):
        if X[i,5]=='convenient':
            X[i,5]=='cconvenient'
        if X[i,4]=='critical':                   
            X[i,4]=='ccritical'

    Xtrain = np.zeros((train.shape[0],27))
    Xtest = np.zeros((test.shape[0],27))
    
    Ytrain = np.zeros((train.shape[0],5))
    
    dictfory = { 'not_recom':0, 'recommend' :1, 'very_recom':2, 'priority':3, 'spec_prior':4 }
    dictop = { 'usual':0, 'pretentious':1, 'great_pret':2, 'proper':3, 'less_proper':4, 'improper':5, 'critical':6, 'very_crit':7, 'complete':8, 'completed':9, 'incomplete':10, 'foster':11, '1':12, '2':13, '3':14, 'more':15, 'convenient':16, 'less_conv':17, 'ccritical':18, 'cconvenient':19, 'inconv':20, 'nonprob':21, 'slightly_prob':22, 'problematic':23, 'recommended': 24, 'priority':25, 'not_recom':26}
    dict = {}
    for i in range(train.shape[1]):
        dict[i] = {}
    dict[0]['usual'] = 0
    dict[0]['pretentious'] = 1
    dict[0]['great_pret'] = 2
    dict[1]['proper'] = 3
    dict[1]['less_proper'] = 4
    dict[1]['improper'] = 5
    dict[1]['critical'] = 6
    dict[1]['very_crit'] = 7
    dict[2]['complete'] = 8
    dict[2]['completed'] = 9
    dict[2]['incomplete'] = 10
    dict[2]['foster'] = 11
    dict[3]['1'] = 12
    dict[3]['2'] = 13
    dict[3]['3'] = 14
    dict[3]['more'] = 15
    dict[4]['convenient'] = 16
    dict[4]['less_conv'] = 17
    dict[4]['critical'] = 18
    dict[5]['convenient'] = 19
    dict[5]['inconv'] = 20
    dict[6]['nonprob'] = 21
    dict[6]['slightly_prob'] = 22
    dict[6]['problematic'] = 23
    dict[7]['recommended'] = 24
    dict[7]['priority'] = 25
    dict[7]['not_recom'] = 26
    dict[8]['very_recom'] = 2
    dict[8]['spec_prior'] = 4
    dict[8]['recommend'] = 1
    dict[8]['priority'] = 3
    dict[8]['not_recom'] = 0
    for i in range(0,train.shape[0]):
        Ytrain[i,dict[8][X[i,8]]]=1
        for j in range(0,8):
            Xtrain[i,dict[j][X[i,j]]]=1
    for i in range(0,test.shape[0]):
        for j in range(0,8):
            Xtest[i,dict[j][XX[i,j]]]=1 
            
    Xtrain = np.insert(Xtrain,0,1,axis=1)
    Xtest = np.insert(Xtest,0,1,axis=1)
    
    f = open(paramsinput)
    params=[]
    for ss in f.read().split():
        params.append((ss))
    
    if params[0]=='1' :
        learningrate = float(params[1])
        maxiter = int(params[2])
        batch = int(params[3])
        
        # print(learningrate,maxiter,batch)
        
        weights = np.zeros((28,5))
        for p in range(0,maxiter):
            for r in range(0,int(Xtrain.shape[0]/batch)):
                if (r==int(Xtrain.shape[0]/batch)-1):
                    Xtrainbat = Xtrain[r*batch:,:]
                    Ytrainbat = Ytrain[r*batch:,:]
                else :
                    Xtrainbat = Xtrain[r*batch:(r+1)*batch,:]
                    Ytrainbat = Ytrain[r*batch:(r+1)*batch,:]
                
                weights = weights + (learningrate/Xtrainbat.shape[0])*error(Xtrainbat,Ytrainbat,predict(Xtrainbat,weights))
        predictions = predict(Xtest,weights) 
        print(LA.norm(weights))
        pred = np.argmax(predictions,axis=1)
        preder = []
        for rt in range(0,test.shape[0]):
            if pred[rt]==0:
                preder.append('not_recom')
            elif pred[rt]==1:
                preder.append('recommend')
            elif pred[rt]==2:
                preder.append('very_recom')
            elif pred[rt]==3:
                preder.append('priority')
            elif pred[rt]==4:
                preder.append('spec_prior')     
        
        np.savetxt(sys.argv[4],preder,delimiter=',',fmt='%s')
        np.savetxt(sys.argv[5],[[p for p in line] for line in weights],delimiter=',',fmt='%s')
        
        
    if params[0]=='2' :
        seeds = float(params[1])
        maxiter = int(params[2])
        batch = int(params[3])
        weights = np.zeros((28,5))
        
        for p in range(0,maxiter):
            for r in range(0,int(Xtrain.shape[0]/batch)):
                if (r==int(Xtrain.shape[0]/batch)-1):
                    Xtrainbat = Xtrain[r*batch:,:]
                    Ytrainbat = Ytrain[r*batch:,:]
                else :
                    Xtrainbat = Xtrain[r*batch:(r+1)*batch,:]
                    Ytrainbat = Ytrain[r*batch:(r+1)*batch,:]
                learningrate = seeds/math.sqrt(p+1)    
                weights = weights + (learningrate/Xtrainbat.shape[0])*error(Xtrainbat,Ytrainbat,predict(Xtrainbat,weights))
       
        predictions = predict(Xtest,weights)    
        pred = np.argmax(predictions,axis=1)
        preder = []
        for rt in range(0,test.shape[0]):
            if pred[rt]==0:
                preder.append('not_recom')
            elif pred[rt]==1:
                preder.append('recommend')
            elif pred[rt]==2:
                preder.append('very_recom')
            elif pred[rt]==3:
                preder.append('priority')
            elif pred[rt]==4:
                preder.append('spec_prior')     
        
        np.savetxt(sys.argv[4],preder,delimiter=',',fmt='%s')
        np.savetxt(sys.argv[5],[[p for p in line] for line in weights],delimiter=',',fmt='%s')  
        
        
    if params[0]=='3' :
        initial, alpha, beta = map(float,params[1].split(','))
        maxiter = int(params[2])
        batch = int(params[3])
        weights = np.zeros((28,5))
        learningrate = initial
        for p in range(1,maxiter+1):
            while entropy(weights+learningrate*error(Xtrain,Ytrain,predict(Xtrain,weights)),Xtrain,Ytrain) > entropy(weights,Xtrain,Ytrain) + learningrate*alpha*np.power(LA.norm(error(Xtrain,Ytrain,predict(Xtrain,weights)),2),2):
                learningrate *= beta
            weights = weights + (learningrate/train.shape[0])*error(Xtrain,Ytrain,predict(Xtrain,weights))
        predictions = predict(Xtest,weights)    
        pred = np.argmax(predictions,axis=1)
        preder = []
        for rt in range(0,test.shape[0]):
            if pred[rt]==0:
                preder.append('not_recom')
            elif pred[rt]==1:
                preder.append('recommend')
            elif pred[rt]==2:
                preder.append('very_recom')
            elif pred[rt]==3:
                preder.append('priority')
            elif pred[rt]==4:
                preder.append('spec_prior')     
        
        np.savetxt(sys.argv[4],preder,delimiter=',',fmt='%s')
        np.savetxt(sys.argv[5],[[p for p in line] for line in weights],delimiter=',',fmt='%s')      
            
            
if __name__ == '__main__':
    part1(*sys.argv[1:])         
