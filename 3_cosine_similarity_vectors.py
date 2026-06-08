import numpy as np 

def cosine_similarity_1d(A,B):
    mag_A = np.linalg.norm(A)
    mag_B = np.linalg.norm(B)
    denom = mag_A * mag_B
    if denom == 0:
        return 0.0
    return np.dot(A,B) / denom
   

def cosine_similarity_batched(A,B):
    mag_A = np.linalg.norm(A,axis=1,keepdims=True) # shape (3,1)
    mag_B = np.linalg.norm(B,axis=1,keepdims=True) # shape
    print(mag_A ) # shape (3,1)
    print(mag_B)        # shape (4,1)
    denom = mag_A @ mag_B.T # shape (3,1) x (1,4) = (3,4)
    #cant do this as denom is not scalar 
    # if denom == 0:
    #     return 0.0
    # np.where checks every element individually across the whole matrix, so it works naturally on arrays.
    return np.where(denom == 0, 0.0, A@B.T/ denom)


#given
if __name__ == "__main__":
    A = np.array([[1,2],[3,4],[5,6]]) # shape (3,2)
    B = np.array([[1,1],[2,3],[4,5],[6,7]]) # shape(4,2)
    # print(cosine_similarity_1d(A,B.T)) # cant broad cast as 3,2 * 4,2 not possible, B.T is 2,4 so AB = 3,2, x 2,4 = 3 X 4 
    
    print(cosine_similarity_batched(A,B)) # vector siz 3* 4