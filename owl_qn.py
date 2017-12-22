import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *
from adadelta import *
from sklearn import metrics

# import warnings
# warnings.filterwarnings("error")

train_path = pardir+'/data/train'
test_path = pardir+'/data/test'
e = 1e-6
lamda = 0.0001#batch 100

def sigmoid(z): 
    return 1/(1+np.exp(-z))

def compute_regular_gradients(x,y,w):
    alpha = 1
    h = sigmoid(x*w)
    m = np.shape(x)[0]
    return alpha*x.T*(h-y)/m

def get_pesudo_gradient(gfunc,x,y,w,isl1=1):
    if not isl1:
        return gfunc(x,y,w)
    g = [[0.0]]*len(w)
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
                g[i]=[0.0]
    g = np.matrix(g)
    return g
    
def get_orthant(w,psudo_g):
    orthant = np.copy(w)
    nozeroindex = (w!=0)
    zeroindex = (w==0)
    nozerosign = np.sign(orthant[nozeroindex])
    if len(nozerosign)>0:
        orthant[nozeroindex]=nozerosign
    zerosign = np.sign(-psudo_g[zeroindex])
    if len(zerosign)>0:
        orthant[zeroindex]=zerosign.A1
    return orthant

def fix_sign(g,sign):
    res = np.multiply(g,sign)
    g[res<=0]=0.0
    return g
    
def computeloss(y,x,w):
    h = sigmoid(x*w)
    return -(np.dot(y.T,np.log(h))+np.dot((1-y).T,np.log(1-h)))+lamda*np.sum(np.abs(w))
    
def owl_qn(func,gfun,gpfun,w,trainx,trainy):
    delta = 0.3
    be = 0.5
    epsilo = 1e-4
    e = 1e-10
   
    k = 0
    m = 10
    s = []
    y = []
    while k<10:
        g = gpfun(gfun,trainx,trainy,w)
        qk = g.copy()
        t = len(s)
        i = t-1
        if t>0:
            a = []
            while i>=0:
                alpha = s[i].T*qk/(y[i].T*s[i]+e)
                a.append(alpha)
                qk -= y[i]*alpha
                i-=1
            if t>=1:
                qk*=s[-1].T*y[-1]/(y[-1].T*y[-1]+e)
            for i in range(t):
                beta = y[i].T*qk/(y[i].T*s[i]+e)
                qk += s[i]*(a[t-1-i]-beta)
            if y[-1].T*s[-1]>0:
                d =-qk
                if np.dot(d.T,-g)<=0:
                    print("error")
                d = fix_sign(d,-g)
            else:
                print("less than zero")
        else:
            d =-qk
            if np.dot(d.T,-g)<=0:
                print("error")
            d = fix_sign(d,-g)
        #line search 
        z = 0
        orth = get_orthant(w,g)
        
        # if d>=0:
            # print("error: no descent gradient")
            # return 
        while z<20:
            new_w = w+be**z*d
            new_w = fix_sign(new_w,orth) 
            temp1 = func(trainy,trainx,new_w)
            temp2 = func(trainy,trainx,w)+delta*be**z*g.T*(new_w-w)
            # temp2 = func(trainy,trainx,w)-delta*be**z*g.T*d
            if temp1<=temp2:
                break
            z+=1 

        if len(s)>m:
            s.pop(0)
            y.pop(0)
            
        sk = new_w - w
        oldg = gfun(trainx,trainy,w)
        w = new_w.copy()
        newg = gfun(trainx,trainy,w)
        yk = newg-oldg  
        s.append(sk)
        y.append(yk)  
        k+=1
        # if k%10==0:
        test(w)
    return w   

def online_owl_qn(func,gfun,gpfun,w,trainx,trainy):
    epsilo = 1e-4
    n = np.shape(trainx)[1]

    c = 1
    k = 0
    m = 10
    s = []
    y = []
    minimum = 1
    batch_size = 1
    locallamda = 0.1
    lr = 0.01
    t0 =np.power(10,4)
    samples = np.shape(trainx)[0]
    ada = Adam(n,alpha=0.1)
    while k<10:
        print("current iter:" +str(k))
        indexs = list(range(samples))
        np.random.shuffle(indexs)
        lastj=0
        for j in range(len(indexs)):
            if j%batch_size!=0 or j==0:
                continue
            c_batch = j/batch_size
            g = gpfun(gfun,trainx[lastj:j,:],trainy[lastj:j,:],w)
            qk = g.copy()
            t = len(s)
            if t>0:
                i = t-1
                a = []
                while i>=0:
                    alpha = s[i].T*qk/(y[i].T*s[i]+e)
                    a.append(alpha)
                    qk -= y[i]*alpha
                    i-=1
                if t>0:
                    # qk*=s[0].T*y[0]/(y[0].T*y[0]+e)
                    qk*=s[-1].T*y[-1]/(y[-1].T*y[-1]+e)
                for i in range(t):
                    beta = y[i].T*qk/(y[i].T*s[i]+e)
                    qk += s[i]*(a[t-1-i]-beta)
                if y[-1].T*s[-1]>0:
                    d =-qk
                    # d = fix_sign(d,-g)
            else:
                d = -qk
                # d = fix_sign(d,-g)
            # if np.dot(g.T,d)>0:
                # print("error: no descent gradient")
                # return 
            templr = ada.getgrad(d,c_batch)
            templr = fix_sign(templr,-g)
            if np.dot(templr.T,-g)<0:
                print("error")
            # templr = lr*d
            # if np.sum(templr)>0:
                # print("error: 1 no descent gradient")
            new_w = w+templr
            orth = get_orthant(w,g)
            new_w = fix_sign(new_w,orth)

            sk = (new_w - w)
            # print(sk[sk==0])
            oldg = gfun(trainx[lastj:j,:],trainy[lastj:j,:],w)
            w = new_w.copy()
            if len(s)>m:
                s.pop(0)
                y.pop(0)

            newg = gfun(trainx[lastj:j,:],trainy[lastj:j,:],w)
            yk = newg-oldg+locallamda*sk
            lastj = j 
            s.append(sk)
            y.append(yk)
                
            if j/batch_size%10 == 0:
                test(w)
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
    # w = np.matrix(np.ones((features+1,1)))
    w  = np.matrix(np.zeros((features,1)))
    # w = np.matrix(np.random.randn(features+1,1))*0.01
    # w =  np.random.normal(0, 1, features+1)
    # w = np.matrix(w.reshape((features+1,1)))
    # ones = np.ones((samples,1))
    # train = np.hstack((ones,train))
    return train,label,w
    
def acc(pa,label):
    p = np.copy(pa)
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
    l = computeloss(label,test,w)/len(label)
    print("loss "+str(l)+" auc "+str(auc(p, label)))
    
    
def train():
    train,label,w = initdata(train_path)
    maxiter = 100
    w = online_owl_qn(computeloss,compute_regular_gradients,get_pesudo_gradient,w,train,label)
    # w = owl_qn(computeloss,compute_regular_gradients,get_pesudo_gradient,w,maxiter,train,label)
    
if __name__=="__main__":
    train()
            