import numpy as np
# from sympy import *
import random
import matplotlib.pylab as plt
from sklearn import linear_model
import math

from adadelta import *
from sklearn import datasets
from sklearn import preprocessing

from getPath import *
pardir = getparentdir()
from commonLib import *
from commonmethod import *
from convertmethod import *

train_path = pardir+'/data/train.ffm'
test_path = pardir+'/data/test.ffm'
auc_path = pardir+'/res/auc_sgd_l2'
e=1e-6


def steepest(w,vecfeatures,labels):
    maxiters = 500
    alpha = 0.01
    for i in range(maxiters):
        g = compute_regular_gradients(vecfeatures,labels,w)
        w = w-g
        if i%5==0:
            test(w)
    
def sgd(w,vecfeatures,labels):
    maxiters = 100
    alpha = 0.1
    samples = len(labels)
    batch_size = 1
    for k in range(maxiters):
        vecfeatures,labels = shufflesamples(vecfeatures,labels)
        lasti = 0
        for i in range(samples):
            if i%batch_size!=0 or i==0:
                continue 
            g = compute_regular_gradients(vecfeatures[lasti:i],labels[lasti:i,:],w)
            w = w-alpha*g
            lasti = i
            if i/batch_size%10==0:
                test(w)
    
def sgd_with_ada(w,vecfeatures,labels):
    if os.path.exists('resp'):
        os.remove('resp')
    maxiters = 10
    samples = len(labels)
    featurenum = np.shape(w)[0]
    ada = Adam(featurenum,alpha=0.01)
    batch_size = 10
    one_size = samples/batch_size
    for k in range(maxiters):
        lasti = 0
        iter = k*one_size
        vecfeatures,labels = shufflesamples(vecfeatures,labels)
        for i in range(samples):
            if i%batch_size!=0 or i==0:
                continue 
            c_batch = i/batch_size
            begin = time.time()
            if not isneg:
                g = compute_regular_gradients(vecfeatures[lasti:i],labels[lasti:i,:],w,isl2=0)
            else:
                g = compute_gradients(vecfeatures[lasti:i],labels[lasti:i,:],w,isl2=0)
            end = time.time()
            print_consume_time(begin,end,"compute_gradients...",isprint=0) 
            templr = ada.getgrad(g,c_batch)
            # w = w-0.001*g
            w -= templr
            # print(g)
            tempw = g[g!=0]
            # line = " ".join(str(x) for x in tempw.A1)+'\n'
            # write_middle_res(line,'resp')
            lasti = i
            if c_batch%10==0:
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
    # sgd_with_ada(w,features,label)
    sgd_with_ada(w,train_features,train_labels)
    

if __name__=="__main__":
    train()
    
    
    
    