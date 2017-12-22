import numpy as np
from commonLib import *
from sklearn import metrics
from getPath import *
pardir = getparentdir()
l2co = 0.01
test_method_path = pardir+'/data/testmethoddata'
isneg = 0

def sparse_feature_w_multiply(featuredic,w):
    wx = 0.0
    keys = np.fromiter(iter(featuredic.keys()), dtype=int)
    iterable = (v for v in featuredic.values())
    values = np.fromiter(iterable, dtype=float)
    return np.sum(w.A1[keys]*values)
    # for k,v in featuredic.items():
        # wx += w.A1[k]*v
    # return wx

def compute_regular_gradients(vecfeatures,labels,w,isl2=0):
    nfeatures = len(w)
    grad = np.matrix(np.zeros((nfeatures,1)))
    begin = time.time()
    # print(vecfeatures[:3])
    temp = sigmoid((np.array([[sparse_feature_w_multiply(vecfeatures[i],w)] for i in range(len(vecfeatures))])))
    end = time.time()
    print_consume_time(begin, end, "sigmoid...")
    temp -= labels
    begin = time.time()
    dic = {}
    lenfeature = len(vecfeatures)
  
    for i in range(lenfeature):
        keys = np.fromiter(iter(vecfeatures[i].keys()), dtype=int)
        iterable = (v for v in vecfeatures[i].values())
        values = np.fromiter(iterable, dtype=float).reshape(len(vecfeatures[i]),-1)
        grad[keys,:]+=temp[i]*values
    # newgrad = np.matrix(np.zeros((nfeatures,1)))  
    
    if isl2:
        grad[:-1,:]+= l2co*w[:-1,:]
    grad /= (lenfeature)
    end = time.time()
    print_consume_time(begin, end, "compute_regular_gradients recursion")
    return grad

def print_consume_time(begin, end, process,isprint=0):
    if isprint:
        print("..."+process+"..."+str((end-begin)))
        
def read_ffm(path):
    with open(path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        features = []
        labels = []
        for line in lines:
            arr = line.split()
            if int(arr[0])==0:
                if isneg:
                    labels.append(-1)
                else:
                    labels.append(0)
            else:
                labels.append(1)
            # labels.appen(int(arr[0]))
            dic ={}
            for a in arr[1:]:
               barr = a.split(':')
               dic[int(barr[1])-1] = float(barr[2])
            features.append(dic)
    features = np.array(features)
    labels = np.matrix(np.array(labels)).T
    return features,labels
    
def get_biggest_dim(features):
    maxfeature = -1
    for featuredic in features:
        temp = np.max(list(featuredic.keys()))
        if temp >= maxfeature:
            maxfeature = temp
    return maxfeature
    
def get_minimum(features):
    minfeature = 10000
    for featuredic in features:
        temp = np.min(list(featuredic.keys()))
        if temp < minfeature:
            minfeature = temp
    return minfeature
   
def update_dic(featuredic,maxfeature):
    featuredic.update({maxfeature+1:1})
    return featuredic

def initdata(path):
    begin = time.time()
    features,labels = read_ffm(path)
    maxfeature = get_biggest_dim(features)
    minfeature = get_minimum(features)
    np.random.seed(1)
    # features = np.array([update_dic(featuredic,maxfeature) for featuredic in features])
    w = np.matrix(np.zeros((maxfeature+1,1)))*1.0
    # w = np.matrix(np.random.randint(2,size = (maxfeature+2,1)))
    # w = np.matrix(np.random.uniform(-1,1,size = (maxfeature+2,1)))
    # print(w)
    # w = np.matrix(np.random.randn(maxfeature+2,1))
    # w = np.zeros((maxfeature+2,1))
    # w = np.matrix([[0] if i%2 else [1] for i in range(maxfeature+2)])
    end = time.time()
    print_consume_time(begin,end,"init data "+path)
    return features,labels,w

def acc(pa,label):
    p = np.copy(pa)
    p[p>=0.5]=1
    if isneg:
        p[p<0.5]=-1
    else:
        p[p<0.5]=0
    p = np.array(p)
    label = np.array(label)
    p = np.squeeze(p)#squeeze pass array not matrix
    label = np.squeeze(label)
    return len(p[p==label])/len(p)
    
def predict(features,w):
    res = np.array([sparse_feature_w_multiply(features[i],w) for i in range(len(features))])
    res = sigmoid(res)
    return res
    
def computeloss(pa,labels,w,isl1=0,l1co=1,isl2=0,l2co=1):
    eps = 1e-15
    p = np.copy(pa)
    p = np.clip(p, eps, 1 - eps)
    if isl2:
        return (-(np.dot(labels.T,np.log(p))+np.dot((1-labels).T,np.log(1-p)))+l2co*w.T*w)/np.shape(labels)[0] 
    if isl1:
        return (-(np.dot(labels.T,np.log(p))+np.dot((1-labels).T,np.log(1-p)))+l1co*np.sum(np.abs(w)))/np.shape(labels)[0] 
    return -(np.dot(labels.T,np.log(p))+np.dot((1-labels).T,np.log(1-p)))/np.shape(labels)[0]
        
        
    
def computeloss_lib(p,labels):
    return metrics.log_loss(labels, p)
    
def sigmoid(z): 
    # z[z>50] = 50
    # z[z<-50] = -50
    # z = np.clip(z, -50, 50)
    # double ex = pow(2.718281828, fres);
    # return ex / (1.0 + ex);
    temp = np.power(2.71828,z)
    return temp*1.0/(1+temp)
    
def comp_loss_with_features(features,labels,w):
    return computeloss(predict(features,w),labels,w)
    
def shufflesamples(vecfeatures,labels):
    indexs = list(range(len(labels)))
    np.random.seed(1)
    np.random.shuffle(indexs)
    return vecfeatures[indexs],labels[indexs]
    
def lbfgs_two_recursion(s,y,newg,d,c=1):
    a = []
    ts = len(s)
    p = ts-1
    e = 1e-10
    while p>=0:
        alpha = s[p].T*newg/(y[p].T*s[p]+e)
        a.append(alpha)
        newg -= y[p]*alpha
        p-=1
    if ts>0:
        # temp = ts-2
        # newg *= s[0].T*y[0]/(y[0].T*y[0]+e)
        newg*=s[-1].T*y[-1]/(y[-1].T*y[-1]+e)
        # newg *= s[temp].T*y[temp]/(y[temp].T*y[temp])
        # g *= s[t].T*y[t]/(y[t].T*y[t])
    for p in range(ts):
        beta = y[p].T*newg/(y[p].T*s[p]+e)
        newg += s[p]*(c*a[ts-1-p]-beta)
    if y[-1].T*s[-1]>0:
        d = -newg
    else:
        print("lesszero")
    return d

def test(w,features,labels,auc_path,isl1=0,istrain=0):
    begin = time.time()
    p = predict(features,w)
    # print(p[:10])
    end = time.time()
    print_consume_time(begin,end,"predict",isprint=0)
    loss = computeloss(p,labels,w,isl1=isl1)
    end1 = time.time()
    print_consume_time(end,end1,"computeloss",isprint=0) 
    if istrain:
        lines = "train acc:"+str(acc(p,labels))+" auc:"+str(cal_auc(p, labels))+" loss:"+str(loss.A1[0])+'\n'
    else:
        lines = "test acc:"+str(acc(p,labels))+" auc:"+str(cal_auc(p, labels))+" loss:"+str(loss.A1[0])+'\n'
        print(lines)
        write_middle_res(lines,auc_path)
    
def get_updates(w,update):
    a = np.linalg.norm(w)
    update_value = np.linalg.norm(update)
    print(update_value/a)
    
    
if __name__=="__main__":
    # features,labels,w = initdata(test_method_path)
    # grad = compute_regular_gradients(features,labels,w)
    # print(grad)
    print(sigmoid([1,2]))