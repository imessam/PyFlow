import numpy as np

class Optimizer:
    pass

class SGD(Optimizer):
    
    def __init__(self,alpha):
        self.alpha=alpha
    
    def update(self,parameters, grads):
        """
        Update parameters using gradient descent
    
        Arguments:
        parameters -- python dictionary containing your parameters 
        grads -- python dictionary containing your gradients, output of backward
    
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
        return parameters