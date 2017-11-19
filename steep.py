import numpy as np
# from sympy import *
import random
import matplotlib.pylab as plt
from sklearn import linear_model
import math
from getPath import *
pardir = getparentdir()
from commonLib import *

train_path = pardir+'/data/train'
test_path = pardir+'/data/test'

def sigmoid(z):
    # print(z)
    return 1/(1+np.exp(-z))

def compute_gradients(x,y,w,alpha):
    return alpha*x.T*(sigmoid(x*w)-y)
    
def compute_regular_gradients(x,y,w,alpha,lamda,isl1):
    h = sigmoid(x*w)
    m = np.shape(x)[0]
    if isl1==0:
        return alpha*x.T*(h-y)/m
    elif isl1==1:
        return alpha*(x.T*(h-y)+lamda*np.sign(w))/m
    elif isl1==2:
        return alpha*(x.T*(h-y)+lamda*w)/m
    elif isl1==3:
        return alpha*(x.T*(h-y)+lamda*w+lamda*np.sign(w))/m

def steepest(x,y,w,isl1):
    maxiters = 1000
    alpha = 0.1
    lamda = 1
    i = 0
    loss = 1000
    for i in range(maxiters):
        g = compute_regular_gradients(x,y,w,alpha,lamda,isl1)
        w = w-g
        if i%5==0:
            lastloss = loss
            loss = computeloss(y,w,x)/np.shape(x)[0]
            # print("loss"+str(loss))
            t = np.abs(lastloss-loss)
            print(t)
            if t<1e-4:
                break
            # print(np.linalg.norm(g))
        v = np.linalg.norm(g)
        # if i%100==0:
            # print(v)
        # if v<1e-3:
            # break
    return w
    
def computeloss(y,w,x):
    h = sigmoid(x*w)
    return -(np.dot(y.T,np.log(h))+np.dot((1-y).T,np.log(1-h)))
    
def sgd(x,y,w,isl1):
    maxiters = 20
    alpha = 0.01
    lamda = 1
    error = []
    samples = np.shape(x)[0]
    count=0
    loss = 10000
    for i in range(maxiters):
        indexs = list(range(samples))
        np.random.shuffle(indexs)
        lastw = w
        lastp = sigmoid(x*w)
        lasta = acc(lastp,y)

        for index in indexs:
            # p = sigmoid(np.sum(w*x[index,:]))
            # p = round(p)
            # if p == y[index]:
                # continue
            g = compute_regular_gradients(np.array(x[index,:]),y[index],w,alpha,lamda,isl1)
            w = w-g
            if index==indexs[len(indexs)-1]:
                # print(w)
                # print(g)
                # p = sigmoid(x*w)
                # p[p>0.5]=1
                # p[p<0.5]=0
                # print("loss"+str(np.mean(np.abs(p-y))))
                lastloss = loss
                loss = computeloss(y,w,x)/samples
                # print(loss[0][0])
                t = np.abs(lastloss-loss)
                print(t)
        if t<1e-4:
            break
                # print(w)
                # print("loss"+
        # p = sigmoid(x*w)
        # a = acc(p,y)
        # print(a)
        # dis = np.linalg.norm(lastp-p)/samples#must divide
        # dis = np.linalg.norm(g)
        # print("dis:"+str(dis))
        # if a-lasta<0:
            # count+=1
        # else:
            # count=0
        # if count>5:
            # break
    return w

def initdata(path):
    data = read_dic(path)
    np.random.shuffle(data)
    train = data[:,:-1]
    label = data[:,-1]
    train = np.matrix(np.array(train))
    label = np.matrix(np.array(label)).T
    train = np.matrix(np.array(train))
    (samples,features) = np.shape(train)
    w = np.matrix(np.zeros((features+1,1)))
    ones = np.ones((samples,1))
    train = np.hstack((ones,train))
    return train,label,w

def train():
    train,label,w = initdata(train_path)
    w = sgd(train,label,w,0)
    # w = steepest(train,label,w,3)
    return w
    
def acc(p,label):
    p[p>0.5]=1
    p[p<0.5]=0
    p = np.array(p)
    label = np.array(label)
    p = np.squeeze(p)#squeeze pass array not matrix
    label = np.squeeze(label)
    return len(p[p==label])/len(p)
    
def test(w):
    test,label,_ = initdata(test_path)
    h = test*w
    p = sigmoid(h)
    # print(p)
    print(acc(p,label))
    
def total():
    w = train()
    test(w)
    
def train_with_sklearn():
    data = read_dic(train_path)
    np.random.shuffle(data)
    train = data[:,:-1]
    label = data[:,-1]
    lr = linear_model.LogisticRegression(verbose=1,solver='lbfgs',max_iter=1000)
    lr.fit(train,label)
    data = read_dic(test_path)
    test = data[:,:-1]
    y = data[:,-1]
    p = lr.predict(test)
    # p[p>0.5]=1
    # p[p<0.5]=0
    print(len(p[p==y])/len(p))

if __name__=="__main__":
    # train()
    # train_with_sklearn()
    total()

    
    
    
    