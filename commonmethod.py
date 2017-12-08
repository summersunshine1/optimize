import numpy as np
from commonLib import *
from sklearn import metrics

l2co = 1

def sparse_feature_w_multiply(featuredic,w):
    wx = 0.0
    for k,v in featuredic.items():
        wx += w.A1[k]*v
    return wx

def compute_regular_gradients(vecfeatures,labels,w,isl2=0):
    nfeatures = len(w)
    grad = np.matrix(np.zeros((nfeatures,1)))
    begin = time.time()
    temp = sigmoid(np.array([[sparse_feature_w_multiply(vecfeatures[i],w)] for i in range(len(vecfeatures))]))
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
            labels.append(int(arr[0]))
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
   
def update_dic(featuredic,maxfeature):
    featuredic.update({maxfeature+1:1})
    return featuredic

def initdata(path):
    begin = time.time()
    features,labels = read_ffm(path)
    maxfeature = get_biggest_dim(features)
    np.random.seed(1)
    features = np.array([update_dic(featuredic,maxfeature) for featuredic in features])
    # w = np.matrix(np.zeros((maxfeature+1,1)))
    w = np.matrix(np.random.randn(maxfeature+2,1))
    end = time.time()
    print_consume_time(begin,end,"init data "+path)
    return features,labels,w

def acc(p,label):
    p[p>=0.5]=1
    p[p<0.5]=0
    p = np.array(p)
    label = np.array(label)
    p = np.squeeze(p)#squeeze pass array not matrix
    label = np.squeeze(label)
    return len(p[p==label])/len(p)
    
def predict(features,w):
    res = sigmoid(np.array([sparse_feature_w_multiply(features[i],w) for i in range(len(features))]))
    return res
    
def computeloss(p,labels):
    eps = 1e-15
    p = np.clip(p, eps, 1 - eps)
    return -(np.dot(labels.T,np.log(p))+np.dot((1-labels).T,np.log(1-p)))/np.shape(labels)[0]
    
def computeloss_lib(p,labels):
    return metrics.log_loss(labels, p)
    
def sigmoid(z): 
    z[z>50] = 50
    z[z<-50] = -50
    return 1/(1+np.exp(-z))
    
def comp_loss_with_features(features,labels,w):
    return computeloss(predict(features,w),labels)
    
def shufflesamples(vecfeatures,labels):
    indexs = list(range(len(labels)))
    return vecfeatures[indexs],labels[indexs]
    
