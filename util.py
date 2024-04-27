import numpy as np

def softmax(a):
    ea = np.exp(a - np.max(a, axis=1, keepdims=True))
    return ea / np.sum(ea, axis=1, keepdims=True)

def accuracy(y_true, y_pred):
    return np.sum(np.argmax(y_true,axis=1) == np.argmax(y_pred, axis=1)) / y_true.shape[0]

def toHotEncoding(t, k):
    ans = np.zeros((t.shape[0], k))
    ans[np.arange(t.shape[0]), np.reshape(t, t.shape[0]).astype(int)] = 1
    return ans

def loadDataset(filename):
    """ Read CSV data from file.
    Returns X, Y values with hot targets. """
    labels=open(filename).readline().split(',')
    data=np.loadtxt(filename, delimiter=',')
    X=data[:,:-1] # observations
    T=data[:,-1]  # targets, discrete categorical features (integers)
    K=1 + np.max(T).astype(int) # number of categories
    N=X.shape[0]  # number of observations
    D=X.shape[1]  # number of features per observation
    return X, T
