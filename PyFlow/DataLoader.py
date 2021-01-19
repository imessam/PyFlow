import numpy as np
import pandas as pd

class DataLoader():
    
    def __init__(self,csv_file,normalize=False,shuffle=False):
        self.data=pd.read_csv(csv_file,low_memory=False).values[1:,:]
        self.normalize=normalize
        self.shuffle=shuffle

    def read(self):
        self.data.head()
        
    
    def split(self,X,ratio=1):
        d=np.array(X,dtype=np.float64)[1:,:]
        M=d.shape[0]
        if(self.shuffle):
            np.random.shuffle(d)
        trainLen=int(M*ratio)
        train=d[:trainLen,:]
        test=d[trainLen:,:]
        return train,test
    
    def norm(self,X):
        d=X
        M=d.shape[0]
        mean=np.sum(d,0)/M
        var=np.sum((d-mean)**2,0)/M
        d=(d-mean)/(np.sqrt(var))
        return d
   
    def __call__(self):
        train,test=self.split(self.data,0.6)
        trainX=train[:,:-1]
        trainY=train[:,-1]
        testX=test[:,:-1]
        testY=test[:,-1]    
        if(self.normalize):
            trainX=self.norm(trainX)
            testX=self.norm(testX)
        return trainX,trainY.reshape((trainY.shape[0],1)),testX,testY.reshape((testY.shape[0],1))