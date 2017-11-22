import numpy as np
import random
import matplotlib.pylab as plt
from sklearn import linear_model
from getPath import *
pardir = getparentdir()
from commonLib import *
from sklearn.datasets import make_classification

train_path = pardir+'/data/train'
test_path = pardir+'/data/test'

def sigmoid(z): 
    return 1/(1+np.exp(-z))

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
    # a = np.random.randint(1,10, size=n)
    h = np.eye(n)
   
    k = 0
    m = 5
    s = []
    y = []
    g = gfun(trainx,trainy,w)
    d = -h*g
    while k<maxiter:
        g = gfun(trainx,trainy,w)
        t = np.linalg.norm(g)
        print(t)
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
        print(z)
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
        i = t-2
        a = []
        while i>=0:
            alpha = s[i].T*qk/(y[i].T*s[i])
            a.append(alpha)
            qk = qk-y[i]*alpha
            i-=1
        # if t>=2:
            # h = y[t-1]*s[t-1].T/(y[t-1].T*y[t-1])
            # temp = list(temp.A1)
            # h = np.diag(temp*n)
        r = h*qk
        for i in range(t-1):
            beta = y[i].T*r/(y[i].T*s[i])
            r = r+s[i]*(a[t-2-i]-beta)
        if yk.T*sk>0:
            d = d-r
        k+=1
    return w   
    
def online_lbfgs(func,gfun,hess,w,maxiter,trainx,trainy):
    delta = 0.3
    be = 0.4
    epsilo = 1e-4
    n = np.shape(trainx)[1]
    # a = np.random.randint(1,10, size=n)
    minmum = np.power(10,10)#add minimum may fail
    h = np.eye(n)
    c = 0.1
    k = 0
    m = 10
    s = []
    y = []
    g = gfun(trainx,trainy,w)
    d = -h*g
    lamda = 0.1
    lr = 0.001
    t0 =np.power(10,4)
    samples = np.shape(trainx)[0]
    count = 0
    t=0
    while k<maxiter:
        print("iter"+str(k))
        indexs = list(range(samples))
        np.random.shuffle(indexs)
        lasti=0
        for i in range(len(indexs)):
            if i%1000!=0 or i==0:
                continue
                
            g = gfun(trainx[lasti:i,:],trainy[lasti:i,:],w)
            t = np.linalg.norm(g)
            print(t)
            # print(str(len(d))+":"+str(len(d[d>0])))
            w = w+c*lr*t0/(t0+k)*d
            
            if len(s)>m:
                s.pop(0)
                y.pop(0)
                
            sk = c*lr*t0/(t0+k)*d
            qk = gfun(trainx[lasti:i,:],trainy[lasti:i,:],w)
            yk = gfun(trainx[lasti:i,:],trainy[lasti:i,:],w)-g+lamda*sk
            lasti = i 
            s.append(sk)
            y.append(yk)
            t = len(s)
            p = t-2
            a = []
            while p>=0:
                alpha = c*s[p].T*qk/(y[p].T*s[p])
                a.append(alpha)
                qk = qk-y[p]*alpha
                p-=1
            temp = t
            # if temp>=2:#do that will influence convergence 
                # while temp>=1:
                    # if temp==t:
                        # h = y[temp-1]*s[temp-1].T/(y[temp-1].T*y[temp-1])
                    # else:
                        # h+= y[temp-1]*s[temp-1].T/(y[temp-1].T*y[temp-1])
                    # temp-=1
                # h = h/(t-1)   
            r = h*qk
            for p in range(t-1):
                beta = y[p].T*r/(y[p].T*s[p])
                r = r+s[p]*(a[t-2-p]-beta)
            if yk.T*sk>0:
                d = d-r
        # if t<=epsilo:
            # count+=1
            # print(count)
            # if count>3:
                # return w
        # else:
            # count=0    
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
    print(p)
    print(acc(p,label))
    
def train():
    train,label,w = initdata(train_path)
    maxiter = 100
    w = online_lbfgs(computeloss,compute_regular_gradients,hessian,w,maxiter,train,label)
    # print(w)
    test(w)
    
if __name__=="__main__":
    train()
    





    