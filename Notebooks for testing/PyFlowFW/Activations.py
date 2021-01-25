import numpy as np
import PyFlowFW as pf

class Activation():
    pass

class Sigmoid(Activation):
    
    def __call__(self,Z):
        
        
        A = np.array((1/(1+np.exp(-Z))),dtype=np.float64)  
        
        cache = Z
        
        
        A[np.isnan(A)] = 0
        
        return A, cache
    
class Relu(Activation):
    
    def __call__(self,Z):
    
        
        A = np.array((np.maximum(0,Z)),dtype=np.float64)       
        
        cache = Z
        
        A[np.isnan(A)] = 0
    
        return A, cache
    
class Softmax(Activation):
    
    def __call__(self,z):
        
         
        z = z - np.max(z)
        z_exp = np.exp(z)
        z_sum = np.sum(z_exp,axis=0,keepdims=True)
        s = z_exp/(z_sum)
        cache=z
        
        s[np.isnan(s)] = 0
    
        return s , cache
    