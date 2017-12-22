import numpy as np
from getPath import *
pardir = getparentdir()
from commonLib import *
from adadelta import *
from sklearn import metrics
from commonmethod import *

train_path = pardir+'/data/train.ffm'
test_path = pardir+'/data/test.ffm'
auc_path = pardir+'/res/auc_owl_qn_10'
e = 1e-6
lamda = 0.0001

def reshape_arr(arr):
    return np.squeeze(arr).A1
    
def compfunc(v,lamda):
    if v+lamda<0:
        return [v+lamda]
    elif v-lamda>0:
        return [v-lamda]
    return 0
    

def get_pesudo_gradient(realg,w):
    # shape = np.shape(realg)
    g = np.array([[0.0]]*len(w))
    indexs = np.array(list(range(len(w))))
    # lesszero_index = indexs[reshape_arr(w<0)]
    # morezero_index = indexs[reshape_arr(w>0)]
    zero_index = indexs[reshape_arr(w==0)]
    less = reshape_arr((w<0))
    g[less] = realg[less]-lamda
    more = reshape_arr((w>0))
    g[more] = realg[more]+lamda
    
    leftindex = zero_index[reshape_arr((realg[zero_index]<-lamda))]
    rightindex = zero_index[reshape_arr((realg[zero_index]>lamda))]
    g[leftindex] = realg[leftindex]+lamda
    g[rightindex] = realg[rightindex]-lamda
    g = np.matrix(g)
    return g
    
def get_orthant(w,psudo_g):
    orthant = w.copy()
    nozeroindex = (w!=0)
    zeroindex = (w==0)
    nozerosign = np.sign(orthant[nozeroindex])
    if len(nozerosign)>0:
        orthant[nozeroindex]=nozerosign.A1
    zerosign = np.sign(-psudo_g[zeroindex])
    if len(zerosign)>0:
        orthant[zeroindex]=zerosign.A1
    return orthant

def fix_sign(g,sign):
    res = np.multiply(g,sign)
    g[res<=0]=0.0
    return g 

def online_owl_qn(w,vecfeatures,labels):
    epsilo = 1e-4
    n = np.shape(w)[0]
    c = 1
    k = 0
    m = 10
    s = []
    y = []

    batch_size = 100
    local_lamda = 0.1
    minimum = 1e-10
    
    t0 =np.power(10,4)
    ada = Adam(n,alpha=0.01)
    
    iter = 0
    lr = 0.01
    
    while k<100:
        j = 0
        while j+batch_size<=len(labels):
            iter+=1
            # print("current iter:" +str(iter))
            realg = compute_regular_gradients(vecfeatures[j:j+batch_size],labels[j:j+batch_size],w)
            pg = get_pesudo_gradient(realg,w)
            # pg = realg
            if j==0 and k==0:
                d = -pg*minimum
            d = fix_sign(d,-pg)
            templr = ada.getmaxgrad(d,iter)  
            orth = get_orthant(w,pg)
            
            new_w = w+templr
            new_w = fix_sign(new_w,orth)
            sk = (new_w - w)
            
            newg = compute_regular_gradients(vecfeatures[j:j+batch_size],labels[j:j+batch_size,:],new_w)
            yk = newg-realg+local_lamda*sk
            w = new_w.copy()
            if len(s)>m:
                s.pop(0)
                y.pop(0)
            s.append(sk)
            y.append(yk)    

            begin = time.time()
            d = lbfgs_two_recursion(s,y,pg,d,c=1)
            end = time.time()
            print_consume_time(begin,end,"two recursion",isprint=0)
            # d = fix_sign(d,-pg)
            
            j+=batch_size 
            if iter%10 == 0:
                test(w,test_features,test_labels,auc_path,isl1=1)
        k+=1
    return w     

test_features = 0
test_labels = 0 
train_features = 0
train_labels = 0
    
def train():
    if os.path.exists(auc_path):
        os.remove(auc_path)
    global train_features,train_labels
    train_features,train_labels,w = initdata(test_path)

    global test_features,test_labels
    test_features,test_labels,_ = initdata(test_path)
    maxiter = 10
    w = online_owl_qn(w,train_features,train_labels)
    
if __name__=="__main__":
    train()
            