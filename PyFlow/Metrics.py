import numpy as np
import PyFlow as pf



class Metric():
    
    pass

class BinaryAccuracy(Metric):
    
    def __init__(self):
        pass
    
    def __call__(self,AL,Y):
        
        
        m=Y.shape[0]
        p = np.zeros((m,1))
        
        for i in range(0, AL.shape[0]):
            if AL[i,0] >= 0.5:
                p[i,0] = 1
            else:
                p[i,0] = 0
    
        print("Accuracy: "  + str(np.sum((p == Y)/m)))
        
        return p


class MeanSquaredError(Metric):
    
    
    def __init__(self):
        pass
    
    def __call__(self,AL,Y):
        
        
        m=Y.shape[0]
        p = AL
        
        
    
        print("MSE: "  + str(np.sum((p - Y)**2/m)))
        
        return p


class CategoricalAccuracy(Metric):
    
    def __init__(self):
        pass
    
    def __call__(self,AL,Y):
        
        
        m=Y.shape[0]*Y.shape[1]
        p = np.zeros(Y.shape)
        
        
        
        for i in range(0, AL.shape[0]):
            
            maxIndex=np.argmax(AL[i,:])
            p[i,maxIndex]=1
        print("Accuracy: "  + str(np.sum((p == Y)/m)))
        
        return p