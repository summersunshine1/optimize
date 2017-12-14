import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *
from commonmethod import *
from convertmethod import *

train_path = pardir+'/data/train.ffm'
test_path = pardir+'/data/test.ffm'
auc_path = pardir+'/res/auc_ftrl'

def ftrl(vecfeatures,labels):
    l1=1.0
    l2=1.0
    alpha=0.1
    beta=1.0
    maxiter = 10
    feature_dim = np.shape(w)[0]
    z = np.zeros((feature_dim,1))
    n = np.zeros((feature_dim,1))
    batch_size = 100
    samples = len(vecfeatures)
    for k in range(maxiter):
        lastindex = 0
        for index in range(samples):
            if index%batch_size!=0 or index==0:
                continue 
            w = np.matrix([[0] if np.abs(z[i])<=l1 else [(np.sign(z[i])-z[i])/((beta+np.sqrt(n[i]))/alpha + l2)] for i in range(feature_dim)])
            g = compute_regular_gradients(vecfeatures[lastindex:index],labels[lastindex:index,:],w,isl2=0)
            squarg = np.power(g,2)
            sigma = (np.sqrt(n+squarg)-np.sqrt(n))/alpha
            z += g-np.mutiply(sigma, w)
            n += squarg
            lastindex = index
            if i/batch_size%10==0:
                # test(w,train_features,train_labels,1)
                test(w,test_features,test_labels,auc_path,0)
                
test_features = 0
test_labels = 0 
train_features = 0
train_labels = 0

def train():
    if os.path.exists(auc_path):
        os.remove(auc_path)
    global train_features,train_labels
    train_features,train_labels,w = initdata(train_path)
    global test_features,test_labels
    test_features,test_labels,_ = initdata(test_path)
    ftrl(train_features,train_labels)

if __name__=="__main__":
    train()
                

        
        
        
        
        
        
    