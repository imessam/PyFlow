import numpy as np


class Loss():
    pass
    
class LogLikleHoodLoss(Loss):
        
    def sigmoid(self,Z):
    
        A = 1/(1+np.exp(-Z))
        cache = Z
        
        return A
    
    
    def __call__(self,xm,W,yt):
        
        
        xm=np.array(xm)
        W=np.transpose(np.array(W))
        yt=np.array(yt)
        yp=np.dot(xm,W)
        
        yphat=self.sigmoid(yp)
        ypt=np.multiply(yp,yt)
        
        loss=-np.log(np.abs(((yt/2)-0.5)+yphat))
        
        grad=(-yt*xm)/(1+(np.exp(ypt)))
        grad=np.sum(grad,0)/grad.shape[0]
        
        return yphat,loss,grad
    
class SquaredLoss(Loss):
    
    def compute_cost(self,AL, Y):
        """
        Implement the cost function 

        Arguments:
        AL -- probability vector corresponding to your label predictions
        Y -- true "label" vector 
        

        Returns:
        cost -- regression cost cost
        """

        m = Y.shape[0]

        
        cost=(Y-AL)**2
        cost=cost/(m*2)
        
        grad=-(Y-AL)/m
        
        return cost,grad
    
class RegressionLoss(Loss):
    
    def compute_cost(self,AL, Y,W):
        """
        Implement the cost function 

        Arguments:
        AL -- probability vector corresponding to your label predictions
        Y -- true "label" vector 
        W -- 

        Returns:
        cost -- regression cost cost
        """

        m = Y.shape[0]

        
        cost=(Y-AL)**2
        cost=cost+((lamb/2)*(np.dot(np.transpose(W),W)))
        cost=cost/(m*2)
        
        grad=-(Y-AL)/m
        
        return cost,grad
    
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
        
        grad = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
    
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
    
        return cost,grad
    
