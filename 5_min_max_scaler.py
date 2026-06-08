import numpy as np 

# min max scalar
def min_max_scaler(X, range=(0, 1)): # for default range 0,1  for all range 
    print(range)
    min_val = X.min(axis=0) # shape (2,)
    max_val = X.max(axis=0) # shape (2,)
    scale = (range[1] - range[0]) / (max_val - min_val) # shape (2,)
    X_scaled = range[0] + scale * (X - min_val)
    return X_scaled

# using fit and transform method
class MinMaxScaler:
    def __init__(self, range=(0,1)):
        self.range = range
    
    #learns the parameters (min and max) from the data and stores them as attributes of the scaler object. This allows us to apply the same transformation to new data using the same parameters.
    def fit(self,X):
        self.min_val = X.min(axis=0) # shape (2,)
        self.max_val = X.max(axis=0) # shape (2,)
        self.scale = (self.range[1] - self.range[0]) / (self.max_val - self.min_val) # shape (2,)

    #applies transformation to the data using the parameters computed in fit
    def transform(self,X):
        X_scaled = self.range[0] + self.scale * (X - self.min_val)
        return X_scaled

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

def standard_scalar(X ):
    mean = X.mean(axis=0) # shape (2,)
    std = X.std(axis=0) # shape (2,)
    X_scaled = (X - mean) / std
    return X_scaled

class StandardScaler:
    def __init__(self):
        pass

    def fit(self,X):
        self.mean = X.mean(axis=0) # shape (2,)
        self.std = X.std(axis=0) # shape (2,)

    def transform(self,X):
        X_scaled = (X - self.mean) / self.std
        return X_scaled

    def fit_transform(self,X):
        self.fit(X)
        return self.transform(X)

if __name__ == "__main__":
    X = np.array([[1,2],[3,4],[5,6]]) # shape (3,2)
    # print(min_max_scaler(X))    
    # print(standard_scalar(X))
    # scaler = MinMaxScaler(range=(0,1))
    scaler = StandardScaler() # takes no argument
    # scaler.fit(X)
    # print(scaler.transform(X))
    print(scaler.fit_transform(X))



    
