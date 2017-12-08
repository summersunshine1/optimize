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
auc_path = pardir+'/res/auc_100_scale_01'
co = 0.01
isl2 = 1
    
def online_lbfgs(func,gfun,w,maxiter,vecfeatures,labels):
    n = len(w)
    minimum = 1e-10#add minimum may fail
    c =0.1
    m = 10
    s = []
    y = []
    # g = gfun(trainx,trainy,w)
    # d = -h*g
    lamda = 0.1
    lr = 0.01
    t0 =np.power(10,4)
    
    samples = len(labels)
    count = 0
    t=0
    batch_size = 100
    ada = Adam(n,alpha=0.001)
    one_size = samples/batch_size
    k = 0
    oldg = 0
    while k<maxiter:
        lines = "iter"+str(k)+'\n'
        print(lines)
        write_middle_res(lines,auc_path)
        vecfeatures,labels = shufflesamples(vecfeatures,labels)
        lasti=0
        iter = k*one_size
        for i in range(samples):
            if i%batch_size!=0 or i==0:
                continue
            c_batch = i/batch_size
            g = gfun(vecfeatures[lasti:i],labels[lasti:i,:],w,isl2) 
            if lasti==0:
                d = -gfun(vecfeatures,labels,w,isl2)
                # d = -g#*minimum
            
            begin = time.time()
            templr = ada.getgrad(d,iter+c_batch)
            end = time.time()
            print_consume_time(begin,end,"adam")
            # templr = t0/(t0+k)*lr*d
            # w = w+t0/(t0+k)*lr*d
            w += templr/c
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
            p = ts-2
            a = []
            
            begin = time.time()
            while p>=0:
                alpha = s[p].T*newg/(y[p].T*s[p]+e)
                a.append(alpha)
                newg = newg-y[p]*alpha
                p-=1
            if ts>=2:
                temp = ts-2
                # newg *= s[0].T*y[0]/(y[0].T*y[0])
                newg *= s[temp].T*y[temp]/(y[temp].T*y[temp])
                # g *= s[t].T*y[t]/(y[t].T*y[t])
            # if temp>=2:#do that will influence convergence 
                # while temp>=1:
                    # if temp==t:
                        # h = y[temp-1]*s[temp-1].T/(y[temp-1].T*y[temp-1])
                    # else:
                        # h+= y[temp-1]*s[temp-1].T/(y[temp-1].T*y[temp-1])
                    # temp-=1
                # h = h/(ts-1)   
            for p in range(ts-1):
                beta = y[p].T*newg/(y[p].T*s[p]+e)
                newg += s[p]*(c*a[ts-2-p]-beta)
            if yk.T*sk>0:
                d -= newg
            end = time.time() 
            print_consume_time(begin,end,"two recursion")
            if i/batch_size%10==0:
                # test(w,train_features,train_labels,1)
                test(w,test_features,test_labels,0)
        k+=1
    return w   

test_features = 0
test_labels = 0 
train_features = 0
train_labels = 0
 
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
    
def train():
    if os.path.exists(auc_path):
        os.remove(auc_path)
    global train_features,train_labels
    train_features,train_labels,w = initdata(train_path)
    global test_features,test_labels
    test_features,test_labels,_ = initdata(test_path)
    maxiter = 10
    w = online_lbfgs(computeloss,compute_regular_gradients,w,maxiter,train_features,train_labels)
    
if __name__=="__main__":
    train()
    
