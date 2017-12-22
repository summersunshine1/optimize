import numpy as np
import time
from getPath import *
pardir = getparentdir()
from commonLib import *
from commonmethod import *
from convertmethod import *

train_path = pardir+'/data/train.ffm'
test_path = pardir+'/data/test.ffm'
auc_path = pardir+'/res/auc_ftrl'

def ftrl(vecfeatures,labels,w):
    l1=0.0001
    l2=0.1
    alpha=0.1
    beta=1.0
    maxiter = 10
    feature_dim = np.shape(w)[0]
    z = np.zeros((feature_dim,1))
    n = np.zeros((feature_dim,1))
    batch_size = 10
    samples = len(vecfeatures)
    lastw = w
    for k in range(maxiter):
        lastindex = 0
        for index in range(samples):
            if index%batch_size!=0 or index==0:
                continue  
            begin = time.time()
           
            # w[lessindex] = 0
            # w[moreindex] = (np.sign(z[moreindex])*l1-z[moreindex])/((beta+np.sqrt(n[moreindex]))/alpha + l2)
            # w = np.matrix([[0] if np.abs(z[i])<=l1 else [(np.sign(z[i])*l1-z[i])/((beta+np.sqrt(n[i]))/alpha + l2)] for i in range(feature_dim)])
            
            g = compute_regular_gradients(vecfeatures[lastindex:index],labels[lastindex:index,:],w,isl2=1)
            squarg = np.square(g)
            oldn = np.copy(n)
            n += squarg
            sigma = (np.sqrt(n)-np.sqrt(oldn))/alpha
            z += g-np.multiply(sigma, w)
            moreindex = (np.abs(z)>l1)
            lessindex = (np.abs(z)<=l1)
            lastindex = index
            w[lessindex] = 0
            w[moreindex] = (np.sign(z[moreindex])*l1-z[moreindex])/((beta+np.sqrt(n[moreindex]))/alpha + l2)
            # print(w[w!=0])
            end = time.time()
            # print_consume_time(begin,end,"compute gradients...",isprint=1) 
            if index/batch_size%10==0:
                # test(w,train_features,train_labels,1)
                # res = w-lastw
                # print(res[res!=0])
                test(w,test_features,test_labels,auc_path,0)
                # lastw = np.copy(w)
                
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
    ftrl(train_features,train_labels,w)

if __name__=="__main__":
    train()
                

        
        
        
        
        
        
    