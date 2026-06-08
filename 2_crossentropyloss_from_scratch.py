#cross entropy loss from scratch 
import numpy as np 
import random

def log_softmax(logits):
    shifted = logits - np.max(logits, axis = -1, keepdims = True)
    return shifted - np.log(np.sum(np.exp(shifted),axis = -1, keepdims = True))

def crossentropy_loss(logits,labels):
    #labels is the true class 
    N = logits.shape[0] #no of samples
    # print(N)
    log_pred_prob = log_softmax(logits) #matric of size N x no of classes
    # print(log_pred_prob)
    correct_log_probs = log_pred_prob[np.arange(N),labels] #pick s th eproability of the correct class 
    #if N = 4, np.arange(N) = 0,1,2,3 it gives us row, and lables = 0,2,1,0 where 0,1,2 are class gives us which column or correct class to pick i.e x,y coordinates int he matrix prob
    loss = -np.mean(correct_log_probs)
    return loss 

if __name__ == "__main__":
        #correct
        logits = np.array([[2.0,1.0,0.5]])
        labels = np.array([0])


        # logits = np.array([[1.0,1.0,1.0],
        #                    [1.0,1.0,1.0]])
        # labels = np.array([0,2])

        loss = crossentropy_loss(logits,labels)
        print(f"loss: {loss:.6f}")