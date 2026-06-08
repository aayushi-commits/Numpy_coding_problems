#output vector o with real numbers
#exponents for each value ex
#sum of exponents sum_ex
#compute normalizing constant z = 1/sum_ex
#compute probability vector y = o * z


# def softmax(o):
#     ex = [2.71828 ** i for i in o]
#     sum_ex = sum(ex)
#     z = 1 / sum_ex
#     y = [i * z for i in ex]
#     return y

# o = [1.0, 2.0, -3.0,4.6,-0.008]
# y = softmax(o)
# print(y)
# print(sum(y)) # sum adds up to 1 

# #pick the most proabble on eusing argmax # greedy search
# def argmax(y):
#     max_index = 0
#     for i in range(1, len(y)):
#         if y[i] > y[max_index]:
#             max_index = i
#     return max_index

# print(argmax(y)) # index of max 

# #or 
# import numpy as np
# print(np.argmax(y)) # index of max using numpy

o = [1.0, 2.0, -3.0,4.6,10.0,-0.008] # given logits

import numpy as np 
def softmax_naive(x):
    x = np.array(o, dtype = np.float64)
    return np.exp(x)/np.sum(np.exp(x))
# print(softmax_naive(o)) 

def softmax_stable(x):
    x = np.array(o, dtype = np.float64)
    max = np.max(x)
    x_new = x - max # to avoid exploding and shrinking when e of numbers where num>1000 or <-1000
    return np.exp(x_new)/np.sum(np.exp(x_new))
print(softmax_stable(o)) # predicted probabilities for each class ( one correct class)

y = [0,0,1,0,1,0] # one hot encoding of the correct class #actual labels

def cross_entropy_loss ( y,y_pred):
    loss = -np.sum( y * np.log(y_pred))
    return loss

print(cross_entropy_loss(y,softmax_stable(o))) # loss for the predicted probabilities and actual labels