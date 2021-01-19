import numpy as np


class Loss():
    pass

class PerceptronLoss(Loss):
    
    def __call__(self,xm,W,yt):
        
        xm=np.array(xm)
        W=np.transpose(np.array(W))
        yt=np.array(yt)
        yp=np.dot(xm,W)
        ypt=np.multiply(yp,yt)
        
        loss=[max(0,-i) for i in ypt]
        
        grad=np.zeros(xm.shape)
        for i,y in enumerate(yt):
            if loss[i]!=0:
                grad[i,:]=xm[i,:]*y
                
        return yp,loss,grad
    
class PerceptronLossSVM(Loss):
    
    def __call__(self,xm,W,yt):
        
        xm=np.array(xm)
        W=np.transpose(np.array(W))
        yt=np.array(yt)
        yp=np.dot(xm,W)
        ypt=np.multiply(yp,yt)
        
        loss=[max(0,1-i) for i in ypt]
        
        grad=np.zeros(xm.shape)
        for i,y in enumerate(yt):
            if loss[i]!=0:
                grad[i,:]=xm[i,:]*y
                
        return yp,loss,grad
    
class IdentityLoss(Loss):
    
    def __call__(self,xm,W,yt):
        
        xm=np.array(xm)
        W=np.transpose(np.array(W))
        yt=np.array(yt)
        yp=np.dot(xm,W)
        ypt=np.multiply(yp,-yt)
        
        loss=np.log(1+np.exp(ypt))
        
        grad=[]
        
        return yp,loss,grad

    
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
    
    def __call__(self,xm,W,yt):
        
        xm=np.array(xm)
        W=np.transpose(np.array(W))
        yt=np.array(yt)
        yp=np.dot(xm,W)
        
        loss=(yt-yp)**2
        
        grad=[]
        
        return yp,loss,grad
    
class RegressionLoss(Loss):
    
    def __call__(self,xm,W,yt,lamb):
        
        xm=np.array(xm)
        W=np.transpose(np.array(W))
        yt=np.array(yt)
        yp=np.dot(xm,W)
        
        loss=(yt-yp)**2
        loss=loss+((lamb/2)*(np.dot(np.transpose(W),W)))
        loss=loss/(xm.shape[0]*2)
        
        grad=((yt-yp)*xm)
        grad=-np.sum(grad,0)/grad.shape[0]
        
        return yp,loss,grad
    
class CrossEntropyLoss(Loss):
    
    def compute_cost(self,AL, Y):
        """
        Implement the cost function defined by equation (7).

        Arguments:
        AL -- probability vector corresponding to your label predictions, shape (number of examples,1)
        Y -- true "label" vector (for example: containing 0 if non-cat, 1 if cat), shape (number of examples,1)

        Returns:
        cost -- cross-entropy cost
        """
    
        m = Y.shape[0]
        #print(m)
        #print(Y.shape)
        #print(AL.shape)

        # Compute loss from aL and y.
        cost = (1./m) * (-np.dot(Y.T,np.log(AL)) - np.dot((1-Y).T, np.log(1-AL)))
        #print(cost.shape)
    
        cost = np.squeeze(cost)      # To make sure your cost's shape is what we expect (e.g. this turns [[17]] into 17).
        assert(cost.shape == ())
    
        return cost
    
