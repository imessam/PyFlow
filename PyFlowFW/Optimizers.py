import numpy as np

class Optimizer:
    pass

class SGD(Optimizer):
    
    def __init__(self,alpha=0.001):
        self.alpha=alpha
    
    def update(self,parameters, grads):
        """
        Update parameters using stochastic gradient descent
    
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients
    
            Returns:
            parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
            """
    
        L = len(parameters) // 2 # number of layers in the neural network
        #print(L)

        # Update rule for each parameter. Use a for loop.
        for l in range(L):
            
            parameters["W" + str(l+1)] = parameters["W" + str(l+1)] - self.alpha * grads["dW" + str(l+1)]
            
            parameters["b" + str(l+1)] = parameters["b" + str(l+1)] - self.alpha * grads["db" + str(l+1)]
            
            parameters["W" + str(l+1)][np.isnan(parameters["W" + str(l+1)])] = 0
            parameters["b" + str(l+1)][np.isnan(parameters["b" + str(l+1)])] = 0
            
        return parameters
    
    
    
class Adam(Optimizer):
    
    def __init__(self,alpha=0.0001,beta1=0.999,beta2=0.999):
        
        self.alpha=alpha
        self.beta1=beta1
        self.beta2=beta2
        self.i=0
        
        self.init_momentum=True
        
    def init_momentums(self,grads):
        
        
        
        self.m0=grads
        self.m1=grads
        
        for l in range(self.L):
        
            self.m0["dW" + str(l+1)]=np.random.rand(grads["dW" + str(l+1)].shape[0],grads["dW" + str(l+1)].shape[1])
            self.m1["dW" + str(l+1)]=np.random.rand(grads["dW" + str(l+1)].shape[0],grads["dW" + str(l+1)].shape[1])
            
            self.m0["db" + str(l+1)]=np.random.rand(grads["db" + str(l+1)].shape[0],grads["db" + str(l+1)].shape[1])
            self.m1["db" + str(l+1)]=np.random.rand(grads["db" + str(l+1)].shape[0],grads["db" + str(l+1)].shape[1])
        
        self.init_momentum=False
        
    def update(self,parameters, grads):
        
        """
        Update parameters using Adam optimizer
    
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients
    
            Returns:
            parameters -- python dictionary containing your updated parameters 
                  parameters["W" + str(l)] = ... 
                  parameters["b" + str(l)] = ...
            """
        
        self.i=self.i+1
        epsilon=1e-07
        self.L = len(parameters) // 2
        
        if(self.init_momentum):
            self.init_momentums(grads)
        
        m0_hat=self.m0
        m1_hat=self.m1
        
        for l in range(self.L):
        
        
            self.m0["dW" + str(l+1)]=(self.m0["dW" + str(l+1)]*self.beta1) + ((1-self.beta1)*(grads["dW" + str(l+1)]))
            self.m1["dW" + str(l+1)]=(self.m1["dW" + str(l+1)]*self.beta2) + ((1-self.beta2)*(grads["dW" + str(l+1)]**2))
            self.m0["dW" + str(l+1)][np.isnan(self.m0["dW" + str(l+1)])] = 0
            self.m1["dW" + str(l+1)][np.isnan(self.m1["dW" + str(l+1)])] = 0
        
            self.m0["db" + str(l+1)]=(self.m0["db" + str(l+1)]*self.beta1) + ((1-self.beta1)*(grads["db" + str(l+1)]))
            self.m1["db" + str(l+1)]=(self.m1["db" + str(l+1)]*self.beta2) + ((1-self.beta2)*(grads["db" + str(l+1)]**2))
            self.m0["db" + str(l+1)][np.isnan(self.m0["db" + str(l+1)])] = 0
            self.m1["db" + str(l+1)][np.isnan(self.m1["db" + str(l+1)])] = 0
        
            m0_hat["dW" + str(l+1)]=(self.m0["dW" + str(l+1)])/((1-self.beta1)**self.i)
            m1_hat["dW" + str(l+1)]=(self.m1["dW" + str(l+1)])/((1-self.beta2)**self.i)
            m0_hat["dW" + str(l+1)][np.isnan(m0_hat["dW" + str(l+1)])] = 0
            m1_hat["dW" + str(l+1)][np.isnan(m1_hat["dW" + str(l+1)])] = 0
            
            
            m0_hat["db" + str(l+1)]=(self.m0["db" + str(l+1)])/((1-self.beta1)**self.i)
            m1_hat["db" + str(l+1)]=(self.m1["db" + str(l+1)])/((1-self.beta2)**self.i)
            m0_hat["db" + str(l+1)][np.isnan(m0_hat["db" + str(l+1)])] = 0
            m1_hat["db" + str(l+1)][np.isnan(m1_hat["db" + str(l+1)])] = 0
        
            parameters["W" + str(l+1)]=parameters["W" + str(l+1)] - ((self.alpha)*((m0_hat["dW" + str(l+1)])/(np.sqrt(m1_hat["dW" + str(l+1)])+epsilon)))
            parameters["b" + str(l+1)]=parameters["b" + str(l+1)] - ((self.alpha)*((m0_hat["db" + str(l+1)])/(np.sqrt(m1_hat["db" + str(l+1)])+epsilon)))
            parameters["W" + str(l+1)][np.isnan(parameters["W" + str(l+1)])] = 0
            parameters["b" + str(l+1)][np.isnan(parameters["b" + str(l+1)])] = 0
        
        return parameters