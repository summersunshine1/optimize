import numpy as np
import random
import matplotlib.pylab as plt
from sklearn import linear_model
from getPath import *
pardir = getparentdir()
from commonLib import *
from adadelta import *
from sklearn.datasets import make_classification
import time
import datetime

e = 1e-6
train_path = pardir+'/data/train.ffm'
test_path = pardir+'/data/test.ffm'
auc_path = pardir+'/res/auc'

def sparse_feature_w_multiply(featuredic,w):
    wx = 0.0
    for k,v in featuredic.items():
        wx += w.A1[k]*v
    return wx
 
def compute_regular_gradients(vecfeatures,labels,w):
    begin = time.time()
    nfeatures = len(w)
    grad = np.matrix(np.zeros((nfeatures,1)))
    temp = sigmoid(np.array([[sparse_feature_w_multiply(vecfeatures[i],w)] for i in range(len(vecfeatures))]))
    temp -= labels
    for i in range(len(vecfeatures)):
        for k,v in vecfeatures[i].items():
            grad[k,:]+=temp[i]*v
    grad /= (len(vecfeatures))
    end = time.time()
    print_consume_time(begin, end, "compute_regular_gradients")
    return grad
    
def print_consume_time(begin, end, process,isprint=0):
    if isprint:
        print("..."+process+"..."+str((end-begin)))
    
def online_lbfgs(func,gfun,w,maxiter,vecfeatures,labels):
    n = len(w)
    minmum = np.power(10,10)#add minimum may fail
    c =1
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
    ada = Adam(n)
    
    k = 0
    while k<maxiter:
        print("iter"+str(k))
        indexs = list(range(samples))
        np.random.shuffle(indexs)
        lasti=0
        for i in range(len(indexs)-1):
            if i%batch_size!=0 or i==0:
                continue
            
            g = gfun(vecfeatures[lasti:i],labels[lasti:i,:],w) 
            if lasti==0:
                d = -g
            
            begin = time.time()
            templr = ada.getgrad(d,k+1)
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
            yk = gfun(vecfeatures[lasti:i],labels[lasti:i,:],w)-g+lamda*sk

            lasti = i 
            s.append(sk)
            y.append(yk)
            ts = len(s)
            p = ts-2
            a = []
            
            begin = time.time()
            while p>=0:
                alpha = s[p].T*g/(y[p].T*s[p]+e)
                a.append(alpha)
                g = g-y[p]*alpha
                p-=1
            temp = ts
            # if temp>=2:#do that will influence convergence 
                # while temp>=1:
                    # if temp==t:
                        # h = y[temp-1]*s[temp-1].T/(y[temp-1].T*y[temp-1])
                    # else:
                        # h+= y[temp-1]*s[temp-1].T/(y[temp-1].T*y[temp-1])
                    # temp-=1
                # h = h/(ts-1)   
            for p in range(ts-1):
                beta = y[p].T*g/(y[p].T*s[p]+e)
                g += s[p]*(c*a[ts-2-p]-beta)
            if yk.T*sk>0:
                d -= g
            end = time.time() 
            print_consume_time(begin,end,"two recursion")
            # if i/batch_size == 0:
            # g = gfun(vecfeatures,labels,w)/len(vecfeatures)
            # t = np.linalg.norm(g)
            # print("current grad:"+str(t))
            # if i/batch_size%10==0:
            test(w)
            # if t<0.02:
                # return w
        k+=1
    return w   
    
def read_ffm(path):
    with open(path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        features = []
        labels = []
        for line in lines:
            arr = line.split()
            labels.append(int(arr[0]))
            dic ={}
            for a in arr[1:]:
               barr = a.split(':')
               dic[int(barr[1])-1] = float(barr[2])
            features.append(dic)
    features = np.array(features)
    labels = np.matrix(np.array(labels)).T
    return features,labels
    
def get_biggest_dim(features):
    maxfeature = -1
    for featuredic in features:
        temp = np.max(list(featuredic.keys()))
        if temp >= maxfeature:
            maxfeature = temp
    return maxfeature

def initdata(path):
    begin = time.time()
    features,labels = read_ffm(path)
    maxfeature = get_biggest_dim(features)
    np.random.seed(1)
    # w = np.matrix(np.zeros((maxfeature+1,1)))
    w = np.matrix(np.random.randn(maxfeature+1,1))
    end = time.time()
    print_consume_time(begin,end,"init data "+path)
    return features,labels,w

def acc(p,label):
    p[p>=0.5]=1
    p[p<0.5]=0
    p = np.array(p)
    label = np.array(label)
    p = np.squeeze(p)#squeeze pass array not matrix
    label = np.squeeze(label)
    return len(p[p==label])/len(p)
    
def predict(features,w):
    res = sigmoid(np.array([[sparse_feature_w_multiply(features[i],w)] for i in range(len(features))]))
    return res
  
test_features = 0
test_labels = 0

def computeloss(p,labels):
    return -(np.dot(labels.T,np.log(p))+np.dot((1-labels).T,np.log(1-p)))/np.shape(labels)[0]
 
def test(w):
    begin = time.time()
    p = predict(test_features,w)
    loss = computeloss(p,test_labels)
    end = time.time()
    print_consume_time(begin,end,"test") 
    lines = "acc:"+str(acc(p,test_labels))+" auc:"+str(cal_auc(p, test_labels))+" loss:"+str(loss.A1[0])+'\n'
    print(lines)
    write_middle_res(lines,auc_path)
    
def train():
    # train,label,w = initdata(train_path)
    # maxiter = 10
    # w = lbfgs(computeloss,compute_regular_gradients,hessian,w,maxiter,train,label)
    # test(w)
    if os.path.exists(auc_path):
        os.remove(auc_path)
    features,label,w = initdata(train_path)
    global test_features,test_labels
    test_features,test_labels,_ = initdata(test_path)
    maxiter = 10
    w = online_lbfgs(computeloss,compute_regular_gradients,w,maxiter,features,label)
    
if __name__=="__main__":
    train()
    
