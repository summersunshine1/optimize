import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *
from commonmethod import *

train_path = pardir+'/data/train.ffm'
test_path = pardir+'/data/test.ffm'
auc_path = pardir+'/res/auc_sgdlbfgs'

def sgdlbfgs(w,vecfeatures,labels):
    m = 10
    L=10
    k = 0
    t = -1
    maxiter = 10
    batch_size = 100
    samples = len(labels)
    vecfeatures,labels = shufflesamples(vecfeatures,labels)
    lasti = 0
    nfeatures = np.shape(w)[0]
    wavg = np.matrix(np.zeros(nfeatures,1))
    lastavg = 0
    ada = Adam(n,alpha=0.001)
    randommax = samples-batch_size
    s = []
    y = []
    d = 0 
    for k in range(maxiter):
        lasti = i
        i = i+batch_size
        if i>=samples:
            lasti = 0
            i = batch_size
            vecfeatures,labels = shufflesamples(vecfeatures,labels)
        g = gfun(vecfeatures[lasti:i],labels[lasti:i,:],w)
        wavg += w
        if k<=2*L:
            templr = ada.getgrad(g,k+1)
        else:
            templr = ada.getgrad(d,k+1)
        w += templr
        if k%L==0:
            t+=1
            wavg /= L
            if t>0:
                index = np.random.randint(randommax)
                lastindex = index+batch_size
                g1 = gfun(vecfeatures[index:lastindex],labels[index:lastindex],w)
                if d==0:
                    d = -g1
                sk = wavg-lastavg
                yk = g1.T*g1*sk
                if len(s)>m:
                    s.pop(0)
                    y.pop(0) 
                s.append(sk)
                y.append(yk)
            lastavg = np.copy(wavg)
            wavg = np.matrix(np.zeros(nfeatures,1))
            d = lbfgs_two_recursion(s,y,g1,d)
            
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
    w = sgdlbfgs(w,train_features,train_labels)
    
if __name__=="__main__":
    train()
    
                
                
            
            
                
                
        
            
            
            
            
            