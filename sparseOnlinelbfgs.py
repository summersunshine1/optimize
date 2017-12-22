import numpy as np
import random
import matplotlib.pylab as plt
from sklearn import linear_model

from adadelta import *
from sklearn.datasets import make_classification
import time
import datetime


from getPath import *
pardir = getparentdir()
from commonLib import *
from commonmethod import *

e = 1e-6
train_path = pardir+'/data/train.ffm'
test_path = pardir+'/data/test.ffm'
auc_path = pardir+'/res/auc_lbfgs_adagrad_lr_10'
isl2 = 0

    
def online_lbfgs(gfun,w,maxiter,vecfeatures,labels):
    n = len(w)
    minimum = 1e-10#add minimum may fail
    c =1
    m = 10
    s = []
    y = []
    # g = gfun(trainx,trainy,w)
    # d = -h*g
    lamda = 0.1
    lr = 0.1
    t0 =np.power(10,4)
    
    samples = len(labels)
    count = 0
    t=0
    batch_size = 10
    ada = Adam(n,alpha=0.01)
    one_size = samples/batch_size
    k = 0
    oldg = 0
    while k<maxiter:
        lines = "iter"+str(k)+'\n'
        print(lines)
        write_middle_res(lines,auc_path)
        vecfeatures,labels = shufflesamples(vecfeatures,labels)
        lasti=0
        for i in range(samples):
            if i%batch_size!=0 or i==0:
                continue
            c_batch = i/batch_size
            g = gfun(vecfeatures[lasti:i],labels[lasti:i,:],w,isl2) 
            if lasti==0 and k==0:
                # d = -gfun(vecfeatures,labels,w,isl2)
                d = -g*minimum
            
            begin = time.time()
            # templr = ada.getgrad(d,c_batch)
            templr = ada.getmaxgrad(d,c_batch)
            end = time.time()
            print_consume_time(begin,end,"adam")
            # templr = t0/(t0+c_batch)*lr*d
            # get_updates(w,templr)
            # w = w+t0/(t0+k)*lr*d
            w += templr/c 
            # if np.dot(templr.T,-g)<0:
                # print("error")
                # continue
            # print(templr[templr!=0])
            if len(s)>m:
                s.pop(0)
                y.pop(0)   
            sk = templr/c
            newg = gfun(vecfeatures[lasti:i],labels[lasti:i,:],w,isl2)
            yk = newg-g+lamda*sk
            
            lasti = i 
            s.append(sk)
            y.append(yk)
            ts = len(s)
            # if np.dot(templr.T,-newg)<0:
                # print("error")
            begin = time.time()
            d = lbfgs_two_recursion(s,y,newg,d,c)
            end = time.time() 
            print_consume_time(begin,end,"two recursion")
            if i/batch_size%10==0:
                # test(w,train_features,train_labels,1)
                test(w,test_features,test_labels,auc_path,0)
        k+=1
    return w   

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
    maxiter = 10
    w = online_lbfgs(compute_regular_gradients,w,maxiter,train_features,train_labels)
    
if __name__=="__main__":
    train()
    
