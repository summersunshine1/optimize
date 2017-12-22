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
e = 1e-6
train_path = pardir+'/data/train'
test_path = pardir+'/data/test'

def compute_regular_gradients(x,y,w,lamda=1,isl1=0):
    alpha = 1
    h = sigmoid(x*w)
    m = np.shape(x)[0]
    if isl1==0:
        return alpha*x.T*(h-y)/m
    elif isl1==1:
        return alpha*(x.T*(h-y))/m+lamda*np.sign(w)/m
    elif isl1==2:
        t = [[0]]
        t+=list(w[1:])
        t = np.matrix(t)
        # print(t)
        return (alpha*(x.T*(h-y))+lamda*w)/m
    elif isl1==3:
        return alpha*(x.T*(h-y)+lamda*w+lamda*np.sign(w))/m
        
def sigmoid(z): 
    # z[z>50] = 50
    # z[z<-50] = -50
    # z = np.clip(z, -50, 50)
    # double ex = pow(2.718281828, fres);
    # return ex / (1.0 + ex);
    temp = np.power(2.71828,z)
    return temp*1.0/(1+temp)
    
def hessian(x,w,lamda =1,isl1=0):
    m = np.shape(x)[0]
    n = np.shape(x)[1]
    h = sigmoid(x*w)
    temp =  np.multiply(h,(h-1))
    temp = temp.A1
    d = np.diag(temp)
    r = 1/m*x.T*d*x
    if isl1==2:
        r = r+np.diag([0]+[lamda]*(n-1))/m
        # print(np.shape(r))
    return r
    
def computeloss(y,x,w):
    h = sigmoid(x*w)
    return -(np.dot(y.T,np.log(h))+np.dot((1-y).T,np.log(1-h)))
    
def isSemi(b):
    B = np.linalg.eigvals(b)
    if np.all(B>0):
        return True
    return False

def bfgs(func,gfun,hess,w,maxiter,trainx,trainy):
    delta = 0.3
    beta = 0.4
    epsilo = 1e-4
    n = np.shape(trainx)[1]
    # a = np.random.randint(1,10, size=n)
    b = np.eye(n)
    # b = hess(trainx,w)#+np.eye(n)
    # print(isSemi(b))
    k = 0
    
    while k<maxiter:
        g = gfun(trainx,trainy,w)
        t = np.linalg.norm(g)
        if t<=epsilo:
            print("last:"+str(t))
            return w
        # d = -1.0*b*g
        d = -np.linalg.solve(b, g)
        print(d)
        m = 0
        while m<20:
            temp1 = func(trainy,trainx,(w+beta**m*d))
            temp2 = func(trainy,trainx,w)+delta*beta**m*g.T*d
            if temp1<=temp2:
                break
            m+=1 
        print(str(t)+":"+str(m))
        alpha = beta**m
        w = w+alpha*d
        s = alpha*d
        y = gfun(trainx,trainy,w)-g
        if y.T*s>0:
            bs = b*s
            bssb = bs*s.T*b
            sbs = s.T*b*s
            b = b+y*y.T/(y.T*s)-bssb/sbs
            # ys = y.T*s
            # b= (1-s*y.T/(ys))*b*(1-y*s.T/(ys))+s*s.T/(ys)#just shake...
        else:
            print('less')
        k+=1
    return w 
    
def lbfgs(func,gfun,hess,w,maxiter,trainx,trainy):
    delta = 0.3
    be = 0.4
    epsilo = 1e-4
    n = np.shape(trainx)[1]
   
    k = 0
    m = 10
    s = []
    y = []
    g = gfun(trainx,trainy,w)
    d = -g
    while k<100:
        g = gfun(trainx,trainy,w)
        t = np.linalg.norm(g)
        if t<=epsilo:
            print("last"+str(t))
            return w
        z = 0
        while z<20:
            temp1 = func(trainy,trainx,(w+be**z*d))
            temp2 = func(trainy,trainx,w)+delta*be**z*g.T*d
            if temp1<=temp2:
                break
            z+=1 
        w = w+be**z*d
        
        if len(s)>m:
            s.pop(0)
            y.pop(0)
            
        sk = be**z*d
        qk = gfun(trainx,trainy,w)
        yk = gfun(trainx,trainy,w)-g
          
        s.append(sk)
        y.append(yk)
        t = len(s)
        # gamma=1
        # if k>=m:
            # gamma = (s[t-2].T*y[t-2])/(y[t-2].T*y[t-2])

        # qk = qk*gamma
        i = t-1
        a = []
        while i>=0:
            alpha = s[i].T*qk/(y[i].T*s[i])
            a.append(alpha)
            qk = qk-y[i]*alpha
            i-=1
        if t>=1:
            qk*=s[0].T*y[0]/(y[0].T*y[0])
            # temp = list(temp.A1)
            # h = np.diag(temp*n)
        # r = h*qk
        for i in range(t):
            beta = y[i].T*qk/(y[i].T*s[i])
            qk += s[i]*(a[t-1-i]-beta)
        if yk.T*sk>0:
            d =-qk
        k+=1
        if k%10==0:
            test(w)
    return w
    
def onlinebfgs(func,gfun,hess,w,maxiter,trainx,trainy):
    epsilo = 1e-4
    minmum = 1/np.power(10,10)
    n = np.shape(trainx)[1]
    
    samples = np.shape(trainx)[0]
    b = np.eye(n)
    c = 0.1
    batch_size = 1
    lr = 0.001
    lamda = 0.1
    k = 0
    t0 = 10000
    t=1000
    ada = Adam(n)
    count = 0
    while k<maxiter:
        indexs = list(range(samples))
        np.random.shuffle(indexs)
        lasti=0
        for i in range(len(indexs)):
            if i%batch_size!=0 or i==0:
                continue
            g = gfun(trainx[lasti:i,:],trainy[lasti:i,:],w)
            # d = -1.0*b*g
            d = -np.linalg.solve(b, g)
            tempg = ada.getgrad(d,k+1)
            # w = w+t0/(t0+k)*lr/c*d
            # s = t0/(t0+k)*lr/c*d
            w = w+tempg/c
            s = tempg/c
            y = gfun(trainx[lasti:i,:],trainy[lasti:i,:],w)-g+lamda*s
            if y.T*s>0:
                bs = b*s
                bssb = bs*s.T*b
                sbs = s.T*b*s
                b = b+c*y*y.T/(y.T*s)-bssb/sbs
                # ys = y.T*s
                # b= (1-s*y.T/(ys))*b*(1-y*s.T/(ys))+s*s.T/(ys)#just shake...
            else:
                print('less')
            lasti=i
            if i/batch_size%1000 == 0:
                g = gfun(trainx,trainy,w)
                t = np.linalg.norm(g)
                print(t)
                if t<0.02:
                    return w
        k+=1
    return w
    
def online_lbfgs(func,gfun,hess,w,maxiter,trainx,trainy):
    epsilo = 1e-4
    n = np.shape(trainx)[1]
    minmum = np.power(10,10)#add minimum may fail
    h = np.eye(n)
    c =1
    m = 10
    s = []
    y = []
    # g = gfun(trainx,trainy,w)
    # d = -h*g
    lamda = 3
    lr = 0.01
    t0 =np.power(10,4)
    
    samples = np.shape(trainx)[0]
    count = 0
    t=0
    batch_size = 10
    ada = Adam(n,alpha=0.1)
    
    k = 0
    while k<maxiter:
        print("iter"+str(k))
        indexs = list(range(samples))
        np.random.shuffle(indexs)
        lasti=0
        for i in range(len(indexs)-1):
            if i%batch_size!=0 or i==0:
                continue
            g = gfun(trainx[lasti:i,:],trainy[lasti:i,:],w) 
            if lasti==0:
                d = -h*g
            # temp = np.dot(g.T,d)
            templr = ada.getgrad(d,k+1)
            # templr = t0/(t0+k)*lr*d
            # if (np.sum(templr)>=0 and temp>=0):
                # print("error")
            # elif np.sum(templr)>=0 and temp<0:
                # print("adaerror")
          
            # if np.sum(templr)>=0:
                # print("error")
           
            # w = w+t0/(t0+k)*lr*d
            oldw = np.copy(w)
            w = w+templr/c

            if len(s)>m:
                s.pop(0)
                y.pop(0)   
            sk = templr/c
            qk = gfun(trainx[lasti:i,:],trainy[lasti:i,:],w)
            yk = gfun(trainx[lasti:i,:],trainy[lasti:i,:],w)-g+lamda*sk

            lasti = i 
            s.append(sk)
            y.append(yk)
            ts = len(s)
            p = ts-1
            a = []
            while p>=0:
                alpha = s[p].T*qk/(y[p].T*s[p])
                a.append(alpha)
                qk -= y[p]*alpha
                p-=1
            if t>=1:
                # qk*=s[0].T*y[0]/(y[0].T*y[0])
                qk*=s[-1].T*y[-1]/(y[-1].T*y[-1])
                # temp = list(temp.A1)
                # h = np.diag(temp*n)
            # r = h*qk
            for p in range(ts):
                beta = y[p].T*qk/(y[p].T*s[p])
                qk += s[p]*(a[t-1-p]-beta)
            if yk.T*sk>0:
                d =-qk
                
            if i/batch_size%100 == 0:
                test(w)
                get_updates(oldw,templr)
        k+=1
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
    # w = np.matrix(np.random.randn(features+1,1))
    ones = np.ones((samples,1))
    train = np.hstack((ones,train))
    return train,label,w
    
def acc(p,label):
    p[p>0.5]=1
    p[p<0.5]=0
    p = np.array(p)
    print(p[p==1])
    label = np.array(label)
    p = np.squeeze(p)#squeeze pass array not matrix
    label = np.squeeze(label)
    return len(p[p==label])/len(p)
    
    
    
def test(w):
    test,label,_ = initdata(test_path)
    h = test*w
    p = sigmoid(h)
    l = computeloss(label,test,w)/len(label)
    print("loss "+str(l)+" auc "+str(auc(p, label)))
    
def train():
    # train,label,w = initdata(train_path)
    # maxiter = 10
    # w = lbfgs(computeloss,compute_regular_gradients,hessian,w,maxiter,train,label)
    # test(w)
    train,label,w = initdata(train_path)
    maxiter = 10
    # w = lbfgs(computeloss,compute_regular_gradients,hessian,w,maxiter,train,label)
    w = online_lbfgs(computeloss,compute_regular_gradients,hessian,w,maxiter,train,label)
    test(w)
    
def get_updates(w,update):
    a = np.linalg.norm(w)
    update_value = np.linalg.norm(update)
    print(update_value/a)
    
    
if __name__=="__main__":
    train()
    





    