import numpy as np
import PyFlowFW as pf



class Metric():
    
    pass

class BinaryAccuracy(Metric):
    
    def __call__(self,AL,Y):
        
        
        m=Y.shape[0]
        p = np.zeros((m,1))
        
        for i in range(0, AL.shape[0]):
            if AL[i,0] >= 0.5:
                p[i,0] = 1
            else:
                p[i,0] = 0
    
        print("Accuracy: "  + str(np.sum((p == Y)/m)))
        
        return p


class MeanSquaredError(Metric):
    
    
    def __call__(self,AL,Y):
        
        
        m=Y.shape[0]
        p = AL
        
        
    
        print("MSE: "  + str(np.sum((p - Y)**2/m)))
        
        return p


class CategoricalAccuracy(Metric):
    
    
    def __call__(self,AL,Y):
        
        dataLoader=pf.DataLoader.DataLoader()
        
        
        
        m=Y.shape[0]*Y.shape[1]
        p = np.zeros(Y.shape)
        
        
        
        for i in range(0, AL.shape[0]):
            
            maxIndex=np.argmax(AL[i,:])
            p[i,maxIndex]=1
        
        
        p=dataLoader.toGroundTruth(p)
        Y=dataLoader.toGroundTruth(Y)
        
        print("Accuracy: "  + str(np.sum((p == Y)/m)))
        
        return p
    
    
class ConfusionMatrix(Metric):
    
    def __call__(self,true,pred):
        
        m=true.shape[0]
        p = np.zeros((m,1))
        
        trueP=0
        trueN=0
        falseP=0
        falseN=0
        
        for i in range(0, pred.shape[0]):
            
            if pred[i,0] >= 0.5:
                p[i,0] = 1
            else:
                p[i,0] = 0
                
            if (p[i,0] == true[i,0]):
                if (p[i,0]==1):
                    trueP+=1
                else:
                    trueN+=1
            else:
                
                if (p[i,0]==1):
                    falseP+=1
                else:
                    falseN+=1
                
        result=np.array([[trueP,falseN],[falseP,trueN]])
                         
        accuracy=(trueP+trueN)/m
        precision=(trueP)/(trueP+falseP)
        recall=(trueP)/(trueP+falseN)                 
        F1= 2*((precision*recall)/(precision+recall))                

        return result,accuracy,precision,recall,F1