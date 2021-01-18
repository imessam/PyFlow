import numpy as np

class Activation():
    pass

class Sigmoid(Activation):
    
    def __call__(self,Z):
    
        A = 1/(1+np.exp(-Z))
        cache = Z
        
        return A, cache
    
class Relu(Activation):
    
    def __call__(self,Z):
    
        A = np.maximum(0,Z)   
        cache = Z 
    
        return A, cache
    
class Softmax(Activation):
    
    def __call__(self,z):
        axis=np.argmax(z.shape)
        z_exp = np.exp(z)
        z_sum = np.sum(z_exp,axis=axis,keepdims=True)
        s = z_exp/z_sum
        cache=z
    
        return s , cache
    