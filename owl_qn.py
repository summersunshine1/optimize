import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *
from adadelta import *

import warnings
warnings.filterwarnings("error")

train_path = pardir+'/data/train'
test_path = pardir+'/data/test'
e = 1e-6

def sigmoid(z): 
    return 1/(1+np.exp(-z))

def compute_regular_gradients(x,y,w):
    alpha = 1
    h = sigmoid(x*w)
    m = np.shape(x)[0]
    return alpha*x.T*(h-y)/m

def get_pesudo_gradient(gfunc,x,y,w,lamda=0.01,isl1=1):
    if not isl1:
        return gfunc(x,y,w)
    g = [[0]]*len(w)
    m = np.shape(x)[0]
    # lamda = lamda/m
    
    oldg = gfunc(x,y,w)
    for i in range(len(w)):
        v = oldg.item(i)
        wi = w[i]
        # print(str(v+lamda)+" "+str(wi))
        if wi<0:
           g[i] = [v-lamda]
        elif wi>0:
            g[i] = [v+lamda]
        else:
            if v+lamda<0:
                g[i]=[v+lamda]
            elif v-lamda>0:
                g[i]=[v-lamda]
            else:
                g[i]=[0]
    g = np.matrix(g)
    return g
    
def get_orthant(w,psudo_g):
    orthant = np.copy(w)
    nozeroindex = (orthant!=0)
    zeroindex = (orthant==0)
    nozerosign = np.sign(orthant[nozeroindex])
    if len(nozerosign)>0:
        orthant[nozeroindex]=nozerosign
    zerosign = np.sign(-psudo_g[zeroindex])
    if len(zerosign)>0:
        orthant[zeroindex]=zerosign.A1
    return orthant

def fix_sign(g,sign):
    res = np.multiply(g,sign)
    # print(res[res<=0])
    try:
        a = (res<=0)
    except RuntimeWarning:
        print(res[res<=0])
    if len(res[res<=0])==0:
        # print(g)
        return g
    g[res<=0]=0
    return g

def hessian(x,w,lamda =1,isl1=0):
    m = np.shape(x)[0]
    n = np.shape(x)[1]
    h = sigmoid(x*w)
    temp =  np.multiply(h,(h-1))
    temp = temp.A1
    d = np.diag(temp)
    r = 1/m*x.T*d*x
    return r
    
def computeloss(y,x,w,lamda=1):
    h = sigmoid(x*w)
    return -(np.dot(y.T,np.log(h))+np.dot((1-y).T,np.log(1-h)))-lamda*np.sum(np.sign(w))
    
def owl_qn(func,gfun,gpfun,hess,w,maxiter,trainx,trainy):
    delta = 0.3
    be = 0.4
    epsilo = 1e-4
    n = np.shape(trainx)[1]
   
    k = 0
    m = 5
    s = []
    y = []
    hk = np.eye(n)
    h = np.eye(n)
    while k<maxiter:
        g = gpfun(gfun,trainx,trainy,w)
        gk = gfun(trainx,trainy,w)
        t = np.linalg.norm(g)
        print(t)
        if t<=epsilo:
            print("last"+str(t))
            return w
        d = -hk*g
        d = fix_sign(d,-g)
        #line search 
        z = 0
        orth = get_orthant(w,g)
        while z<20:
            new_w = w+be**z*d

            new_w = fix_sign(new_w,orth) 
            temp1 = func(trainy,trainx,new_w)
            temp2 = func(trainy,trainx,w)-delta*g.T*(new_w-w)
            if temp1<=temp2:
                break
            z+=1 
        if len(s)>m:
            s.pop(0)
            y.pop(0)
            
        sk = new_w - w
        w = np.copy(new_w)
        qk = gfun(trainx,trainy,w)
        yk = qk-gk  
        s.append(sk)
        y.append(yk)
        t = len(s)
        i = t-2
        a = []
        while i>=0:
            try:
                alpha = s[i].T*qk/(y[i].T*s[i])
            except RuntimeWarning:
                print(s[i])
                print(y[i])
            a.append(alpha)
            qk = qk-y[i]*alpha
            i-=1
        # if t>=2:
            # temp = s[t-1].T*y[t-1]/(y[t-1].T*y[t-1])
            # temp = list(temp.A1)
            # h = np.diag(temp*n)
        # h = y[0]*s[0].T/(y[0].T*y[0])
        r = h*qk
        for i in range(t-1):
            beta = y[i].T*r/(y[i].T*s[i])
            r = r+s[i]*(a[t-2-i]-beta)
        
        if yk.T*sk>0:
            hk = r*g.I
        k+=1
    return w   

def online_owl_qn(func,gfun,gpfun,hess,w,maxiter,trainx,trainy):
    epsilo = 1e-4
    n = np.shape(trainx)[1]

    c = 1
    k = 0
    m = 10
    s = []
    y = []
    hk = np.eye(n)
    h = np.eye(n)
    
    batch_size = 10
    
    lamda = 0.1
    lr = 0.001
    t0 =np.power(10,4)
    samples = np.shape(trainx)[0]
    ada = Adam(n)
    while k<10:
        print("current iter:" +str(k))
        indexs = list(range(samples))
        np.random.shuffle(indexs)
        lasti=0
        for i in range(len(indexs)):
            if i%batch_size!=0 or i==0:
                continue
            g = gpfun(gfun,trainx[lasti:i,:],trainy[lasti:i,:],w)
            # print(g)
            # return w
            gk = gfun(trainx[lasti:i,:],trainy[lasti:i,:],w)
            d = -hk*g
            # print(d)
            d = fix_sign(d,-g)
            
            templr = ada.getmaxgrad(d,k+1)
            new_w = w+templr
            # print(templr)
            orth = get_orthant(w,g)
            new_w = fix_sign(new_w,orth)
            sk = (new_w - w)
            # new_w = w+sk
            w = np.copy(new_w)
            if len(s)>m:
                s.pop(0)
                y.pop(0)

            qk = gfun(trainx[lasti:i,:],trainy[lasti:i,:],w)
            yk = qk-gk+lamda*sk
            lasti = i  
            s.append(sk)
            y.append(yk)
            t = len(s)
            p = t-2
            a = []
            # print(sk)
            while p>=0:
                try:
                    alpha = s[p].T*qk/((y[p].T)*(s[p])+e)
                except RuntimeWarning:
                    print(s[s!=0])
                    print(y[y!=0])
                    # print("warning")
                a.append(alpha)
                qk = qk-y[p]*alpha
                p-=1
            # if t>=2:
                # temp = s[t-1].T*y[t-1]/(y[t-1].T*y[t-1])
                # temp = list(temp.A1)
                # h = np.diag(temp*n)
            # h = y[0]*s[0].T/(y[0].T*y[0])
            r = h*qk
            for p in range(t-1):
                beta = y[p].T*r/(y[p].T*s[p]+e)
                r = r+s[p]*(c*a[t-2-p]-beta)
            if yk.T*sk>0:
                hk = r*g.I
                
            if i/batch_size%1000 == 0:
                g = gpfun(gfun,trainx,trainy,w)
                grad = np.linalg.norm(g)
                print(grad)
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
    w = np.matrix(np.ones((features+1,1)))
    # w = np.matrix(np.random.randn(features+1,1))*0.01
    # w =  np.random.normal(0, 1, features+1)
    # w = np.matrix(w.reshape((features+1,1)))
    ones = np.ones((samples,1))
    train = np.hstack((ones,train))
    return train,label,w
    
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
    
def train():
    train,label,w = initdata(train_path)
    maxiter = 100
    w = online_owl_qn(computeloss,compute_regular_gradients,get_pesudo_gradient,hessian,w,maxiter,train,label)
    test(w)
    
if __name__=="__main__":
    train()
            