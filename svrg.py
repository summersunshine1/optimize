import numpy as np
import time
from getPath import *
pardir = getparentdir()
from commonLib import *
from commonmethod import *
from adadelta import *

train_path = pardir+'/data/train.ffm'
test_path = pardir+'/data/test.ffm'
auc_path = pardir+'/res/svrg'

def svrg(w,vecfeatures,labels):
    samples = len(labels)
    vecfeatures,labels = shufflesamples(vecfeatures,labels)
    lasti = 0
    i = 0
    nfeatures = np.shape(w)[0]
    wtemp = np.matrix(np.zeros((nfeatures,1)))
    bacth_size = 100
    randommax = samples-bacth_size
    ada = Adam(nfeatures,alpha=0.01)
    for i in range(2):
        index = np.random.randint(randommax)
        g =  compute_regular_gradients(vecfeatures[index:index+bacth_size],labels[index:index+bacth_size,:],w) 
        w -= 0.1*g
    maxiter = int(samples*2/bacth_size)
    for k in range(100):
        fullgrad = compute_regular_gradients(vecfeatures,labels,w)
        wtemp = w.copy()
        
        for i in range(maxiter):
            index = np.random.randint(randommax)
            g1 =  compute_regular_gradients(vecfeatures[index:index+bacth_size],labels[index:index+bacth_size,:],w)
            # print(g1[g1!=0])
            g2 = compute_regular_gradients(vecfeatures[index:index+bacth_size],labels[index:index+bacth_size,:],wtemp)
            partialgrad = g2-(g1-fullgrad)
            # templr = ada.getgrad(partialgrad,i+1) 
            wtemp -= partialgrad
            if i%10==0:
                test(wtemp,test_features,test_labels,auc_path,0)
        w = wtemp.copy() 
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
    w = svrg(w,train_features,train_labels)
    
if __name__=="__main__":
    train()
    
                
                
            
            
                
                
        
            
            
            
            
            