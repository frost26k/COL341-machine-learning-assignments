import sys
import pandas as pd
import numpy as np
import timeit
import math
import re
from collections import Counter
import nltk
from nltk.stem import PorterStemmer 
ps = PorterStemmer()


def mainP(trainin,testin,pred):
    
    stopwords = {'she', 'be', 'above', 'your', 'are', 'just', 'what', 'doesn', 'you', 'while', 'yourself', "weren't", 'having', 'only', 'in', "doesn't", 'don', "won't", "you've", 'them', 'her', 't', "you'll", 're', 'up', 'been', 'how', 'very', 'there', "don't", 'needn', 'hasn', 'wouldn', 'about', 'has', 'myself', 'further', "haven't", 'which', 'doing', 'haven', 'his', 've', 'as', 'same', 'couldn', 's', 'between', 'theirs', 'then', 'such', "aren't", 'being', 'isn', 'if', 'those', 'they', 'were', 'of', 'can', 'why', 'wasn', 'through', 'him', 'at', 'ain', 'i', 'against', 'was', 'where', "needn't", 'that', 'did', 'do', 'until', 'their', 'o', 'when', 'over', 'own', "hadn't", 'yours', 'aren', 'into', 'so', 'nor', 'had', 'didn', "isn't", 'an', "you'd", 'he', 'mustn', 'shan', 'themselves', 'me', "should've", 'out', "wasn't", 'and', 'no', 'other', 'because', "shan't", 'will', 'it', 'again', 'yourselves', 'or', 'should', 'y', "couldn't", 'both', 'during', 'on', 'all', 'who', 'below', "mightn't", "she's", 'our', 'once', 'ours', 'does', 'the', 'weren', 'we', 'from', 'shouldn', 'its', 'mightn', 'whom', 'itself', 'after', "wouldn't", 'few', 'ma', 'll', 'but', "didn't", 'by', 'any', 'hers', 'to', 'now', "mustn't", 'have', 'each', 'down', 'this', 'some', "that'll", 'himself', 'hadn', 'd', 'a', 'than', 'am', "shouldn't", 'with', 'my', 'most', 'm', 'herself', 'under', 'these', "hasn't", 'for', 'off', 'more', "it's", 'too', 'won', "you're", 'before', 'ourselves', 'not', 'here', 'is'}
    traindata = pd.read_csv(trainin)
    
    positiveC = Counter()
    negativeC = Counter()
    
    for i in range(0,traindata.shape[0]):
        traindata['review'][i] = traindata['review'][i].lower()
        traindata['review'][i] = re.split('\W+', traindata['review'][i])
        traindata['review'][i] = list(set(traindata['review'][i]) - set(stopwords))
                
        for gh in range(0,len(traindata['review'][i])):
            traindata['review'][i][gh] = ps.stem(traindata['review'][i][gh])         
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
    positiveR = len(Ytrain[Ytrain==1])/10000
    negativeR = len(Ytrain[Ytrain==0])/10000
    
    testdata = pd.read_csv(testin)
    for i in range(0,testdata.shape[0]):
        testdata['review'][i] = testdata['review'][i].lower()
        testdata['review'][i] = re.split('\W+', testdata['review'][i])
        testdata['review'][i] = list(set(testdata['review'][i]) - set(stopwords))
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