import numpy as np 

#euclidian dist  (a-b)**2 = a**2 + b**2 + 2abcos0
def dist(A,B):
    sq_dist = np.sum(A**2,axis = 1,keepdims = True) + np.sum(B**2,axis=1,keepdims = True) - 2*A@B.T #B get sbroad casted to matches As dim
    return np.sqrt(sq_dist.clip(0))

#given
if __name__ == "__main__":
    A = np.array([[1,2],[3,4],[5,6]]) # shape (3,2)
    B = np.array([[1,1],[2,3],[4,5],[6,7]]) # shape(4,2)
    print(dist(A,B)) # vector siz 3* 4