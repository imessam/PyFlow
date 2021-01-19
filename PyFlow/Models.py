import numpy as np
import PyFlow as pf



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
            param=layer.init_weights(in_shape)
            paramsDict.update(param)
            in_shape=layer.getUnits()
        
        return paramsDict
    
    def forward(self,X,parameters):
        
        A=X
        L=len(self.layers)
        caches=[]
        
        for i,layer in enumerate(self.layers):
            A,cache=layer.forward(A,parameters)
            caches.append(cache)
        
        return A ,caches
    
    def backward(self,AL,Y,caches):
        
        back=pf.Backward.Backward(AL,Y,caches)
        grads=back.backward()
        
        return grads
    
    def update(self,alpha,parameters,grads):
        
        optim=self.optim
        parameters_new=optim.update(parameters,grads)
        
        return parameters_new
        
    def compile(self,optimizer=pf.Optimizers.SGD(0.01),loss=pf.Losses.CrossEntropyLoss(),metric='accuracy'):
        self.optim=optimizer
        self.loss=loss
        self.metric=metric
    
    def fit(self,X,Y,epochs,batches):
        
        self.printLayers()
        paramsDict=self.init_params(X.shape[1])
        X_in=X
        Y_in=Y
        costs=[]
        
        if(batches==0):
            batch_size=1
            batches=X.shape[0]
        else:
            batch_size=int(X.shape[0]/batches)
        
        for epoch in range(epochs):
        
            for i in range(batches):
            
                begin=i*batch_size
                end=begin+batch_size
                if(end>X.shape[0]):
                    end=X.shape[0]
                X_in=X[begin:end,:]
                Y_in=Y[begin:end,:]
                
                A,caches=self.forward(X_in,paramsDict)
                #print(A)
                cost=self.loss.compute_cost(A,Y_in)
                grads=self.backward(A,Y_in,caches)
                params_dict=self.update(0.01,paramsDict,grads)
                
                #if(i%5==0):
                    
                    #print ("Cost after batch %i: %f" %(i, cost))
                
                costs.append(cost)
            
            print ("Cost after epoch %i: %f" %(epoch, cost))

        
        
        return costs
                