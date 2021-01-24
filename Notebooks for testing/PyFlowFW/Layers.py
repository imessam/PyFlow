import numpy as np
import PyFlowFW.Activations as act

class Layer():
    
    pass
        
class Dense(Layer):
    
    
    def __init__(self,num_units,activation,num_layer):
        
        self.num_units=num_units
        self.activation=activation
        self.num_layer=num_layer
    
    def printLayer(self):
        print(f"units : {self.num_units} , activation : {self.activation} , layer no : {self.num_layer}")
        
    def getUnits(self):
        return self.num_units
    
    def init_weights(self,input_units):
        np.random.seed(1)
    
        W = np.random.randn(self.num_units, input_units)*0.01
        b = np.zeros((1,self.num_units))
    
        assert(W.shape == (self.num_units, input_units))
        assert(b.shape == (1,self.num_units))
    
    
        parameters = {f"W{self.num_layer}": W,
                  f"b{self.num_layer}": b}
    
        return parameters
    
    def getActivation(self):
        
        activ=act.Activation()
        
        if self.activation=="relu":
            activ=act.Relu()
        elif self.activation=="sigmoid":
            activ=act.Sigmoid()
        elif self.activation=="softmax":
            activ=act.Softmax()
        else :
            activ=None
            
        return activ
        
    
    def linear_forward(self,A,W,b):
        """ 
        Arguments:
    A -- activations from previous layer (or input data): (number of examples,size of previous layer )
    W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
    b -- bias vector, numpy array of shape (1,size of the current layer)

    Returns:
    Z -- the input of the activation function, also called pre-activation parameter 
    cache -- a python dictionary containing "A", "W" and "b" ; stored for computing the backward pass efficiently
    """
    
        Z = np.dot(A,np.transpose(W)) + b
        
    
        assert(Z.shape == ( A.shape[0],W.shape[0]))
        
        Z[np.isnan(Z)] = 0
        A[np.isnan(A)] = 0
        W[np.isnan(W)] = 0
        b[np.isnan(b)] = 0
        
        cache = (A, W, b)
    
        return Z, cache
    
    def linear_activation_forward(self,A_prev, W, b, activation=act.Relu()):
        """
        Implement the forward propagation for the LINEAR->ACTIVATION layer

        Arguments:
        A_prev -- activations from previous layer (or input data): (number of examples,size of previous layer )
        W -- weights matrix: numpy array of shape (size of current layer, size of previous layer)
        b -- bias vector, numpy array of shape (1,size of the current layer)

        activation -- the activation to be used in this layer

        Returns:
        A -- the output of the activation function, also called the post-activation value 
        cache -- a python dictionary containing "linear_cache" and "activation_cache";
             stored for computing the backward pass efficiently
        """
    
        # Inputs: "A_prev, W, b". Outputs: "A, activation_cache".
        Z, linear_cache = self.linear_forward(A_prev, W, b)
        A, activation_cache = activation(Z)
        
        A[np.isnan(A)] = 0
    
        assert (A.shape == (A_prev.shape[0],W.shape[0]))
        if isinstance(activation,act.Relu):
            
            cache = (linear_cache, activation_cache,"relu")
            
        elif isinstance(activation,act.Sigmoid) :
        
            cache = (linear_cache, activation_cache,"sigmoid")
        
        elif isinstance(activation,act.Softmax) :

            cache = (linear_cache, activation_cache,"softmax")
            
        return A, cache
        
        
    def forward(self,inputs,params):
        
        activ=self.getActivation()
        A,cache=self.linear_activation_forward(inputs,params[f"W{self.num_layer}"],params[f"b{self.num_layer}"],activ)
        
        A[np.isnan(A)] = 0
        
        return A,cache
    
    
    
    
class Dropout(Layer):
    
    def __init__(self,p):
        self.p=p
       
        
    def printLayer(self):
        print(f"propability : {self.p} ")
        
    def forward(self,A):
        
        u = np.random.binomial(1, self.p, size=A.shape) / self.p
        out = A * u
        cache = u
        
        out[np.isnan(out)] = 0
        
        return out, cache
    
    
    
    
    
    