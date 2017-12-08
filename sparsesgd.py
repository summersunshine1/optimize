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

train_path = pardir+'/data/train.ffm'
test_path = pardir+'/data/test.ffm'
auc_path = pardir+'/res/auc_sgd_100'
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
    maxiters = 10
    alpha = 1
    samples = len(labels)
    featurenum = np.shape(w)[0]
    ada = Adam(featurenum)
    batch_size = 100
    one_size = samples/batch_size
    for k in range(maxiters):
        vecfeatures,labels = shufflesamples(vecfeatures,labels)
        lasti = 0
        lines = "iter"+str(k)+'\n'
        print(lines)
        write_middle_res(lines,auc_path)
        iter = k*one_size
        for i in range(samples):
            if i%batch_size!=0 or i==0:
                continue 
            c_batch = i/batch_size
            g = compute_regular_gradients(vecfeatures[lasti:i],labels[lasti:i,:],w)
            templr = ada.getgrad(g,iter+c_batch)
            w = w-templr
            lasti = i
            if i/batch_size%10==0:
                test(w,train_features,train_labels,1)
                test(w,test_features,test_labels,0)
                

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
 
def test(w,features,labels,istrain=0):
    begin = time.time()
    p = predict(features,w)
    end = time.time()
    print_consume_time(begin,end,"predict") 
    loss = computeloss(p,labels)
    end1 = time.time()
    print_consume_time(end,end1,"computeloss") 
    if istrain:
        lines = "train acc:"+str(acc(p,labels))+" auc:"+str(cal_auc(p, labels))+" loss:"+str(loss.A1[0])+'\n'
    else:
        lines = "test acc:"+str(acc(p,labels))+" auc:"+str(cal_auc(p, labels))+" loss:"+str(loss.A1[0])+'\n'
    print(lines)
    write_middle_res(lines,auc_path)
    

if __name__=="__main__":
    train()
    
    
    
    