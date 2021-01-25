import numpy as np
import PyFlowFW.Activations as activ


class Backward():
    
    def __init__(self,AL, Y, caches,grad):
        self.AL=AL
        self.Y=Y
        self.caches=caches
        self.grad=grad
        
        
    def relu_backward(self,dA, cache):
        """
        Implement the backward propagation for a single RELU unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
    
        Z = cache
        dZ = np.array(dA, copy=True) 
    
        dZ[Z <= 0] = 0
    
        assert (dZ.shape == Z.shape)
        
        dZ[np.isnan(dZ)] = 0
    
        return dZ

    def sigmoid_backward(self,dA, cache):
        """
        Implement the backward propagation for a single SIGMOID unit.

        Arguments:
        dA -- post-activation gradient, of any shape
        cache -- 'Z' where we store for computing backward propagation efficiently

        Returns:
        dZ -- Gradient of the cost with respect to Z
        """
        sigmoid=activ.Sigmoid()
        
        Z = cache
    
        s ,__=sigmoid(Z) 
        dZ = dA * s * (1-s)
    
        assert (dZ.shape == Z.shape)
        
        dZ[np.isnan(dZ)] = 0
    
        return dZ
    
    def softmax_backward(self,dA, cache):
        # z, da shapes - (m, n)
        m, n = cache.shape
        Z = cache
        softmax=activ.Softmax()
        s,__=softmax(Z)
        
       
        dz=[]
        for i in range(m):
            
            dz.append(np.dot(dA[i,:],np.diagflat(s[i,:]) - np.dot(s[i,:], s[i,:].T)))
        
        dz=np.array(dz)
        
        assert (dz.shape == s.shape)
        
        dz[np.isnan(dz)] = 0

        return dz


    
    def linear_backward(self,dZ, cache):
        """
        Implement the linear portion of backward propagation for a single layer (layer l)

        Arguments:
        dZ -- Gradient of the cost with respect to the linear output (of current layer l)
        cache -- tuple of values (A_prev, W, b) coming from the forward propagation in the current layer

        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        A_prev, W, b = cache
        m = A_prev.shape[0]

        dW =  np.dot(dZ.T,A_prev)
        db =  np.reshape(np.sum(dZ,axis=0),(1,dZ.shape[1]))
        dA_prev = np.dot(dZ,W)
    
        assert (dA_prev.shape == A_prev.shape)
        assert (dW.shape == W.shape)
        assert (db.shape == b.shape)
        
        
        dW[np.isnan(dW)] = 0
        db[np.isnan(db)] = 0
        dA_prev[np.isnan(dA_prev)] = 0
        
    
        return dA_prev, dW, db
    
    def linear_activation_backward(self,dA, cache):
        
        """
        Implement the backward propagation for the LINEAR->ACTIVATION layer.
    
        Arguments:
        dA -- post-activation gradient for current layer l 
        cache -- tuple of values (linear_cache, activation_cache) we store for computing backward propagation efficiently
        activation -- the activation to be used in this layer, stored as a text string
    
        Returns:
        dA_prev -- Gradient of the cost with respect to the activation (of the previous layer l-1), same shape as A_prev
        dW -- Gradient of the cost with respect to W (current layer l), same shape as W
        db -- Gradient of the cost with respect to b (current layer l), same shape as b
        """
        linear_cache, activation_cache,activation = cache
        
        #print(activation)
    
        if activation == "relu":
            dZ = self.relu_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        elif activation == "sigmoid":
            dZ = self.sigmoid_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
        
        elif activation == "softmax":
            dZ = self.softmax_backward(dA, activation_cache)
            dA_prev, dW, db = self.linear_backward(dZ, linear_cache)
    
        return dA_prev, dW, db
    
    
    def L_model_backward(self,AL, Y, caches,grad):
        """
        Implement the backward propagation for the 
    
        Arguments:
        AL -- probability vector, output of the forward propagation (L_model_forward())
        Y -- true "label" vector (containing 0 if non-cat, 1 if cat)
        caches -- list of caches 
    
        Returns:
        grads -- A dictionary with the gradients
             grads["dA" + str(l)] = ... 
             grads["dW" + str(l)] = ...
             grads["db" + str(l)] = ... 
        """
        grads = {}
        L = len(caches) # the number of layers
        m = AL.shape[0]
        Y = Y.reshape(AL.shape) # after this line, Y is the same shape as AL
    
        # Initializing the backpropagation
        dAL = grad
        
        #dAL[np.isnan(dAL)] = 0
        
        grads["dA"+str(L)]=dAL
    
        current_cache = caches[L-1]
        grads["dA" + str(L-1)], grads["dW" + str(L)], grads["db" + str(L)] = self.linear_activation_backward(dAL, current_cache)
    
        for l in reversed(range(L-1)):
            current_cache = caches[l]
            dA_prev_temp, dW_temp, db_temp = self.linear_activation_backward(grads["dA" + str(l + 1)], current_cache)
            grads["dA" + str(l)] = dA_prev_temp
            grads["dW" + str(l + 1)] = dW_temp
            grads["db" + str(l + 1)] = db_temp

        return grads
    
    
    def backward(self):
        return self.L_model_backward(self.AL,self.Y,self.caches,self.grad)

    