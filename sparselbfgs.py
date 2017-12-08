import numpy as np
import time
from commonmethod import *
from getPath import *
pardir = getparentdir()

isl2 = 0
train_path = pardir+'/data/train.ffm'
test_path = pardir+'/data/test.ffm'
auc_path = pardir+'/res/auc_100_lbfgs'

def lbfgs(func,gfun,w,maxiter,vecfeatures,labels):
    delta = 0.3
    be = 0.4
    epsilo = 1e-4
   
    k = 0
    m = 10
    s = []
    y = []
    while k<15:
        g = gfun(vecfeatures,labels,w)
        if k == 0:
            d = -g
        z = 0
        begin = time.time()
        oldfun = func(vecfeatures,labels,w)
        while z<20:
            temp = be**z*d
            temp1 = func(vecfeatures,labels,(w+temp))
            temp2 = oldfun +delta*g.T*temp
            if temp1<=temp2:
                break
            z+=1 
        print(z)
        end = time.time()
        print_consume_time(begin,end,"...recuresion...",1)
        w = w+be**z*d
        
        
        if len(s)>m:
            s.pop(0)
            y.pop(0)
            
        sk = be**z*d
        newg = gfun(vecfeatures,labels,w)
        yk = newg-g
          
        s.append(sk)
        y.append(yk)
        t = len(s)
     
        i = t-2
        a = []
        while i>=0:
            alpha = s[i].T*newg/(y[i].T*s[i])
            a.append(alpha)
            newg = newg-y[i]*alpha
            i-=1
            
        if t>=2:
            newg *= s[0].T*y[0]/(y[0].T*y[0])
        for i in range(t-1):
            beta = y[i].T*newg/(y[i].T*s[i])
            newg = newg+s[i]*(a[t-2-i]-beta)
        if yk.T*sk>0:
            d -= newg
        k+=1
        test(w,test_features,test_labels,0)
        
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
    w = lbfgs(comp_loss_with_features,compute_regular_gradients,w,maxiter,train_features,train_labels)
    
if __name__=="__main__":
    train()