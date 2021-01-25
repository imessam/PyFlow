import numpy as np


class Loss():
    pass
    
                            ##########Regression Cost Functions#############   
        
class MSE(Loss):
    
    def compute_cost(self,AL, Y):
        """
        Implement the MSE cost function 

        Arguments:
        AL --  vector corresponding to your label predictions
        Y -- true "label" vector 
        

        Returns:
        cost -- MSE regression cost 
        grad -- MSE regression cost gradient
        """

        m = Y.shape[0]

        
        cost=(Y-AL)**2
        cost=cost/(m*2)
        
        grad=-(Y-AL)/m
        
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        cost[np.isnan(cost)] = 0
        grad[np.isnan(grad)] = 0
        
        return cost,grad   
    
    
                                        ##########Classification Cost Functions############# 
    
class BinaryCrossEntropyLoss(Loss):
    
    def compute_cost(self,AL, Y):
        """
        Implement the cost function 

        Arguments:
        AL -- probability vector corresponding to your label predictions
        Y -- true "label" vector 

        Returns:
        cost -- Binary cross-entropy cost
        grad -- Binary cross-entropy gradient
        """
    
        m = Y.shape[0]

        # Compute loss from aL and y.
        cost = (1./m) * (-np.dot(Y.T,np.log(AL)) - np.dot((1-Y).T, np.log(1-AL)))
        
        grad = - (1./m) * (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        cost[np.isnan(cost)] = 0
        grad[np.isnan(grad)] = 0
    
        return cost,grad
    
    
class CategoricalCrossEntropyLoss(Loss):
    
    def compute_cost(self,AL, Y):
        """
        Implement the cost function 

        Arguments:
        AL -- probability vector corresponding to your label predictions
        Y -- true "label" vector 

        Returns:
        cost -- Categorical cross-entropy cost
        grad -- Categorical cross-entropy gradient
        """
    
        m = Y.shape[0]
        

        # Compute loss from aL and y.
        cost = -(1./m) * (Y*np.log(AL))
        cost=np.array(np.sum(cost))
        
        grad = - (1./m) * (np.divide(Y, AL)) 
    
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
        
        cost[np.isnan(cost)] = 0
        grad[np.isnan(grad)] = 0
    
        return cost,grad
    
