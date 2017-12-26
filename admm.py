import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *
train_path = pardir+'/data/train'
test_path = pardir+'/data/test'

def sigmoid(z): 
    z = np.clip(z, -50, 50)
    temp = np.power(2.71828,z)
    return temp*1.0/(1+temp)

def objective(x,y,w,mu):
    m = np.shape(x)[0]
    h = sigmoid(x*w)
    eps = 1e-15
    h = np.clip(h, eps, 1 - eps)
    return (-np.dot(y.T,np.log(h))-np.dot(1-y.T,np.log(1-h)))+m*mu*np.sum(np.abs(w))

def update_objective(x,y,w,u,z,rho):
    h = sigmoid(x*w)
    eps = 1e-15
    h = np.clip(h, eps, 1 - eps)
    return (-np.dot(y.T,np.log(h))-np.dot(1-y.T,np.log(1-h)))+rho/2*np.sum(np.square(w-z+u))
    
def compute_regular_gradients(x,y,w):
    h = sigmoid(x*w)
    m = np.shape(x)[0]
    return x.T*(h-y)
    
def hessian(x,w):
    m = np.shape(x)[0]
    h = sigmoid(x*w)
    temp =  np.multiply(h,(h-1))
    temp = temp.A1
    d = np.diag(temp)
    r = 1/m*x.T*d*x
    return r
    
def update_w(x,y,w,u,z,rho,maxiter,epsilo=1e-3):
    alpha = 0.1
    beta = 0.5
    m = np.shape(x)[0]
    for k in range(maxiter):
        g = (compute_regular_gradients(x,y,w)+rho*(w-z+u))/m
        # h = hessian(x,w)
        # dx = -np.linalg.solve(h, g)
        dx = -g
        t = np.linalg.norm(g)
        if t<epsilo:
            break
        t = 0
        while(t<20):
            temp1 = update_objective(x,y,w,u,z,rho)
            w += alpha*beta**t*dx
            temp2 = update_objective(x,y,w,u,z,rho)
            if temp2<temp1+alpha*beta**t*g.T*dx:
                break
            t+=1
    return w 

def shrinkage(a,kappa):
    temp = np.array(np.zeros(np.shape(a)))
    return np.multiply(np.sign(a),np.maximum(temp,np.abs(a)-kappa))
        
def admm(x,y,w,u,z,rho,maxiter,mu):
    m = np.shape(x)[0]
    for k in range(maxiter):
        w = update_w(x,y,w,u,z,rho,maxiter)
        alpha = 1
        what = alpha*w+(1-alpha)*z
        zold = z.copy()
        
        z = what + u
        z = shrinkage(z,mu/rho)
        u += what-z
        print(mu/rho)
        ob = objective(x,y,w,mu)/m
        rnorm = np.sum(np.abs(w-z));
        # print(z-zold)
        snorm = np.sum(rho*(np.abs(z-zold)))
        # if k%10==0:
        print("current iter:"+str(k)+" "+str(ob)+"primal norm "+str(rnorm)+" dual norm "+str(snorm))
        test(w,mu)
            
            
def initdata(path):
    data = read_dic(path)
    np.random.shuffle(data)
    train = data[:,:-1]
    label = data[:,-1]
    train = np.matrix(np.array(train))
    label = np.matrix(np.array(label)).T
    train = np.matrix(np.array(train))
    (samples,features) = np.shape(train)
    w = np.matrix(np.zeros((features,1)))
    u = np.matrix(np.zeros((features,1)))
    z = np.matrix(np.zeros((features,1)))
    # w = np.matrix(np.random.randn(features+1,1))
    # ones = np.ones((samples,1))
    # train = np.hstack((ones,train))
    return train,label,w, u ,z
    
def test(w,mu):
    testdata,label,_,_,_= initdata(test_path)
    h = testdata*w
    p = sigmoid(h)
    l = objective(testdata,label,w,mu)/len(label)
    print("loss "+str(l)+" auc "+str(auc(p, label)))
    
def train():
    train,label,w,u,z = initdata(train_path)
    maxiter = 100
    rho = 0.1
    mu = 0.01
    admm(train,label,w,u,z,rho,maxiter,mu)
    
    
if __name__=="__main__":
    train()
            
        
        
    
        
    
    
    