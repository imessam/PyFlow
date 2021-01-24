import numpy as np
import PyFlowFW as pf
import matplotlib.pyplot as plt

class Model():
    pass


class Sequential(Model):
    
    def __init__(self,*args,**kwargs):
        
        self.layers=args[0]
        self.parameters={}
    
    def printLayers(self):
        
        for layer in self.layers:
            layer.printLayer()
    
    def init_params(self,input_shape):
        
        paramsDict={}
        in_shape=input_shape
        
        for i,layer in enumerate(self.layers):
            
            if(isinstance(layer,pf.Layers.Dense)):
                param=layer.init_weights(in_shape)
                paramsDict.update(param)
                in_shape=layer.getUnits()
        
        return paramsDict
    
    def forward(self,X,parameters):
        
        A=X
        L=len(self.layers)
        caches=[]
        
        for i,layer in enumerate(self.layers):
            if(isinstance(layer,pf.Layers.Dense)):
                A,cache=layer.forward(A,parameters)
                caches.append(cache)
            elif(isinstance(layer,pf.Layers.Dropout)):
                A,cache=layer.forward(A)
        
        return A ,caches
    
    def backward(self,AL,Y,caches,grad):
        
        back=pf.Backward.Backward(AL,Y,caches,grad)
        grads=back.backward()
        
        return grads
    
    def update(self,alpha,parameters,grads):
        
        optim=self.optim
        parameters_new=optim.update(parameters,grads)
        
        return parameters_new
        
    def compile(self,optimizer=pf.Optimizers.SGD(0.01),loss=pf.Losses.BinaryCrossEntropyLoss(),metric=None):
        
        self.optim=optimizer
        self.loss=loss
        self.metric=metric
    
    def fit(self,X,Y,epochs,batches,validation_split=0.0):
        
        X_train=X[0:int((1-validation_split)*X.shape[0]),:]
        Y_train=Y[0:int((1-validation_split)*Y.shape[0]),:]
        
        X_valid=X[int((1-validation_split)*X.shape[0]):,:]
        Y_valid=Y[int((1-validation_split)*Y.shape[0]):,:]
        
        print(X_valid.shape[0])
        print(X_train.shape[0])
        
        self.printLayers()
        paramsDict=self.init_params(X.shape[1])
        X_in=X_train
        Y_in=Y_train
        costs=[]
        
        if(batches==0):
            batch_size=1
            batches=X_train.shape[0]
        else:
            batch_size=int(X_train.shape[0]/batches)
        
        for epoch in range(epochs):
            
        
            for i in range(batches):
            
                begin=i*batch_size
                end=begin+batch_size
                if(end>X_train.shape[0]):
                    end=X_train.shape[0]
                X_in=X_train[begin:end,:]
                Y_in=Y_train[begin:end,:]
                
                A,caches=self.forward(X_in,paramsDict)
                cost,grad=self.loss.compute_cost(A,Y_in)
                grads=self.backward(A,Y_in,caches,grad)
                paramsDict=self.update(0.01,paramsDict,grads)
            

               
                
            
            print ("Training loss after epoch %i: %f" %(epoch, cost))
            pred=self.metric(A,Y_in)
            costs.append(cost)
            
            if not(validation_split==0.0):
                A_valid,__=self.forward(X_valid,paramsDict)
                print(f"Validation metric :")
                self.metric(A_valid,Y_valid)


        self.parameters=paramsDict
        
        plt.plot([i for i in range(epochs)],costs)
        plt.xlabel("epochs")
        plt.ylabel("cost")
        
        return costs
    
    def predict(self,X,K):
        """
        This function is used to predict the results of a  L-layer neural network.
    
        Arguments:
        X -- data set of examples you would like to label
        K -- num of units in output layer
    
        Returns:
        p -- predictions for the given dataset X
        """
    
        m = X.shape[0]
        n = len(self.parameters) // 2 # number of layers in the neural network
        p = np.zeros((m,K))
        # Forward propagation
        probas, caches = self.forward(X, self.parameters)

    
        # convert probas to 0/1 predictions
        for i in range(0, probas.shape[0]):
            maxIndex=np.argmax(probas[i,:])
            p[i,maxIndex]=1
    
        #print results
        #print ("predictions: " + str(p))
        #print ("true labels: " + str(y))
        #print("Accuracy: "  + str(np.sum((p == y)/m)))
        
        return probas,p
                