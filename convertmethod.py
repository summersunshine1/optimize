import numpy as np
from commonmethod import *
    
def compute_gradients(vecfeatures,labels,w,isl2=0):
    nfeatures = len(w)
    grad = np.matrix(np.zeros((nfeatures,1)))
    l = labels.A1
    temp = sigmoid(np.array([-sparse_feature_w_multiply(vecfeatures[i],w)*l[i] for i in range(len(vecfeatures))]))
    temp = np.multiply(temp,-l)
    lenfeature = len(vecfeatures)
    for i in range(lenfeature):
        keys = np.fromiter(iter(vecfeatures[i].keys()), dtype=int)
        iterable = (v for v in vecfeatures[i].values())
        values = np.fromiter(iterable, dtype=float).reshape(len(vecfeatures[i]),-1)
        grad[keys,:]+=temp[i]*values
    if isl2:
        grad[:-1,:]+= l2co*w[:-1,:]  
    grad /= (lenfeature)
    return grad
   
def cost(vecfeatures, labels, w,isl2 = 0):
    if isl2:
        return (np.sum(-np.log(sigmoid(sparse_feature_w_multiply(vecfeatures[i],w)*labels[i])) for i in range(len(vecfeatures)))
        +0.5*l2co*w.T*w)/len(vecfeatures)
    return (np.sum(-np.log(sigmoid(sparse_feature_w_multiply(vecfeatures[i],w)*labels[i])) for i in range(len(vecfeatures))))/len(vecfeatures)
        
    

    