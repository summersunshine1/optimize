import numpy as np
# from sympy import *
import random
import matplotlib.pylab as plt
from sklearn import linear_model
import math
from getPath import *
pardir = getparentdir()
from commonLib import *
from adadelta import *
from sklearn import datasets
from sklearn import preprocessing
from sklearn import metrics

train_path = pardir+'/data/train'
test_path = pardir+'/data/test'
e=1e-6

def sigmoid(z):
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
    maxiters = 500
    alpha = 0.01
    lamda = 1
    i = 0
    loss = 1000
    for i in range(maxiters):
        g = compute_regular_gradients(x,y,w,alpha,lamda,isl1)
        w = w-g
        if i%5==0:
            lastloss = loss
            loss = computeloss(y,w,x)/np.shape(x)[0]
            t = np.abs(lastloss-loss)
            print(t)
    return w
    
def computeloss(y,w,x):
    h = sigmoid(x*w)
    return -(np.dot(y.T,np.log(h+e))+np.dot((1-y).T,np.log(1-h+e)))
    
def sgd(x,y,w,isl1):
    maxiters = 100
    alpha = 0.01
    lamda = 1
    error = []
    samples = np.shape(x)[0]
    count=0
    loss = 10000
    for i in range(maxiters):
        indexs = list(range(samples))
        np.random.shuffle(indexs)

        for index in indexs:
            g = compute_regular_gradients(np.array(x[index,:]),y[index],w,alpha,lamda,isl1)
            w = w-g
            if index==indexs[len(indexs)-1]:
                # lastloss = loss
                # loss = computeloss(y,w,x)
                # t = np.abs(lastloss-loss)
                tg = compute_regular_gradients(np.array(x),y,w,alpha,lamda,isl1)
                t = np.linalg.norm(tg)
                print(t)
                if t<1e-4:
                    break
    return w
    
def sgd_with_ada(x,y,w,isl1):
    maxiters = 10
    alpha = 1
    lamda = 1
    error = []
    samples = np.shape(x)[0]
    features = np.shape(x)[1]
    loss = 10000
    ada = Adam(features)
    for i in range(maxiters):
        indexs = list(range(samples))
        np.random.shuffle(indexs)
        for index in indexs:
            g = compute_regular_gradients(np.array(x[index,:]),y[index],w,alpha,lamda,isl1)
            templr = ada.getgrad(g,i+1)
            w = w-templr
            # loss = computeloss(y,w,x)
            # print(loss)
            if index==indexs[len(indexs)-1]:
                tg = compute_regular_gradients(np.array(x),y,w,alpha,lamda,isl1)
                t = np.linalg.norm(tg)
                print(t)
                if t<1e-3:
                    break
    return w
    
xtest=0
ytest=0

def initdata(path,useStandard = 0):
    global xtest,y_test
    if not useStandard:
        data = read_dic(path)
        np.random.shuffle(data)
        train = data[:,:-1]
        label = data[:,-1]
       
    else:
        iris = datasets.load_breast_cancer()
        train = np.array(iris.data)
        label = np.array(iris.target)
        scaler = preprocessing.StandardScaler()
        train, xtest, label, y_test = train_test_split(train, label, test_size=0.2, random_state=42)
        train = scaler.fit_transform(train)
        xtest = scaler.transform(xtest)
    
        xtest = np.matrix(np.array(xtest))
        y_test = np.matrix(np.array(y_test)).T
        (samples,features) = np.shape(xtest)
        ones = np.ones((samples,1))
        xtest = np.hstack((ones,xtest))
        
    train = np.matrix(np.array(train))
    label = np.matrix(np.array(label)).T
    (samples,features) = np.shape(train)
    w = np.matrix(np.zeros((features+1,1)))
    ones = np.ones((samples,1))
    train = np.hstack((ones,train))

    return train,label,w

def train():
    train,label,w= initdata(train_path)
    # w = sgd(train,label,w,0)
    # w = steepest(train,label,w,0)
    w = sgd_with_ada(train,label,w,0)
    return w
    
def acc(p,label):
    p[p>0.5]=1
    p[p<0.5]=0
    p = np.array(p)
    label = np.array(label)
    p = np.squeeze(p)#squeeze pass array not matrix
    label = np.squeeze(label)
    return len(p[p==label])/len(p)
    
def test(w,useStandard=0):
    if not useStandard:
        xtest,y_test,_ = initdata(test_path)
    h = xtest*w
    p = sigmoid(h)
    print(acc(p,y_test))
    # fpr, tpr, thresholds = metrics.roc_curve(y_test.A1, p, pos_label=1)
    # print(metrics.auc(fpr, tpr))
    cal_auc(p, y_test.A1)
    
    
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
    
def train_logistic():
    # iris = datasets.load_breast_cancer()
    # train = np.array(iris.data[:, [1,2]])
    # label = np.array(iris.target)
    # train, xtest, label, y_test = train_test_split(train, label, test_size=0.2, random_state=42)
    # print(train)
    train,label,w= initdata(train_path)
    # print(train)
    lr = linear_model.LogisticRegression(verbose=0)
    lr.fit(train,label)
    xtest,y_test,_ = initdata(test_path)
    p = lr.predict(xtest)
    fpr, tpr, thresholds = metrics.roc_curve(y_test.A1, p, pos_label=1)
    print(metrics.auc(fpr, tpr))
    print(len(p[p==y_test.A1])/len(p))

if __name__=="__main__":
    # train()
    # train_with_sklearn()
    total()
    # train_logistic()
    
    
    
    