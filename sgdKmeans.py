import numpy as np
import random
from sklearn import datasets
import matplotlib.pylab as plt
from kmeans_init import *
from adadelta import *
from bfgs import *

def initialize(x,k):
    samples = np.shape(x)[0]
    features = np.shape(x)[1]
    w = np.zeros((k,features))
    for i in range(features):
        min = np.min(x[:,i])
        max = np.max(x[:,i])
        w[:,i] = min+(max-min)*np.random.rand(k)
    return w

def dis(w,x):
    return np.square(w-x)
    
def kmeans(x,k):
    # w = initialize(x,k)
    w = kmeans_plus(x,k)
    # w = kmeans_plus_improve(x,k)
    samples = np.shape(x)[0]
    features = np.shape(x)[1]
    clusters = np.array([-1]*samples)
    iters = 0
    print("batch kmeans begin train.......")
    while True:
        print("current iter"+str(iters))
        lastclusters = np.copy(clusters)
        lastw = np.copy(w)
        for i in range(samples):
            dist = dis(w,x[i,:])
            tempk = np.argmin(np.sum(dist,1))
            clusters[i] = tempk
        for i in range(k):
            indexs = np.array(range(samples))
            kindexs = indexs[clusters==i]
            w[i,:] = np.mean(x[kindexs,:],0)
        wchange = np.linalg.norm(w-lastw)/k
        print(wchange)
        if wchange<1e-3:
            break
        iters+=1   
    return w,clusters
    
def sgd_kmeans_loss(x,y,w):
    tempk = np.argmin(np.sum(dis(w,x),1)) #get cluster nearest 
    return 1/2*np.square(x-w[tempk,:])-y
    
def sgd_kmeans_hess(x,w):
    (k,features) = np.shape(w)
    grad = np.zeros((k,k,features))
    tempk = np.argmin(np.sum(dis(w,x),1))
    grad[tempk,:,:]=np.ones((k,features))
    grad[:,tempk,:]=np.ones((k,features))
    grad = np.matrix(grad)
    return grad
    
def sgd_kmeans_grad(x,y,w):
    (k,features) = np.shape(w)
    grad = np.zeros((k,features))
    tempk = np.argmin(np.sum(dis(w,x),1))
    grad[tempk,:] = w[tempk,:]-x
    grad = np.matrix(grad)
    return grad
 
def sgd_kmeans(x,k):
    # w = initialize(x,k)
    w = kmeans_plus(x,k)
    # w = kmeans_plus_improve(x,k)
    maxiter = 10
    samples = np.shape(x)[0]
    features = np.shape(x)[1]

    clusters = np.array([-1]*samples)
    
    ada = Adam(k,features)
    print("sgd kmeans begin train.......")
    for iter in range(maxiter):
        print("current iter"+str(iter))
        indexs = list(range(samples))
        np.random.shuffle(indexs)
        lastw = np.copy(w)
        clusternum = np.array([0]*k)
        for i in indexs:
            tempk = np.argmin(np.sum(dis(w,x[i,:]),1)) #get cluster nearest 
            clusternum[tempk]+=1 #nk=nk+1
            # w[tempk,:]+=1/(clusternum[tempk])*(x[i,]-w[tempk,:])#w = w+1/n*(xi-wk)
            grad = np.zeros((k,features))
            temp = 1/(clusternum[tempk])*(x[i,:]-w[tempk,:])
            grad[tempk,:] = temp
            w = w+ada.getmaxgrad(grad,iter+1)
        wchange = np.linalg.norm(w-lastw)/k
        print(wchange)
        if wchange<1e-2:
            break
    for i in range(samples):
        dist = dis(w,x[i,:])
        tempk = np.argmin(np.sum(dist,1))
        clusters[i] = tempk        
    return w,clusters
    
def getdata():
    iris = datasets.load_iris()
    x = np.array(iris.data[:, [1,2]])
    y = np.array(iris.target)
    return x,y
    
def sgd_kmeans_with_fake_label(k=3):
    x,y = getdata()
    features = np.shape(x)[1]
    samples = np.shape(x)[0]
    w = initialize(x,k)
    y = np.zeros((samples,1))
    x = np.matrix(x)
    y = np.matrix(y)
    w = np.matrix(w)
    w = onlinebfgs(sgd_kmeans_loss,sgd_kmeans_grad,sgd_kmeans_hess,w,10,x,y)
    clusters = np.array([-1]*samples)
    for i in range(samples):
        dist = dis(w,x[i,:])
        tempk = np.argmin(np.sum(dist,1))
        clusters[i] = tempk
    return w,clusters

def kmeans_train():
    x,y = getdata()
    colors = ['b','g','r']
    indexs = np.array(range(len(y)))
    plt.subplot(131)    #nrows, ncols, plot_number
    for i in range(3):
        temp = indexs[y==i]
        plt.scatter(x[temp,0],x[temp,1],color = colors[i])
    w,clusters = kmeans(x,3)
    plt.subplot(132)
    for i in range(3):
        temp1 = indexs[clusters==i]
        plt.scatter(x[temp1,0],x[temp1,1],color = colors[i])
    w,clusters = sgd_kmeans_with_fake_label(3)    
    plt.subplot(133)
    for i in range(3):
        temp2 = indexs[clusters==i]
        plt.scatter(x[temp2,0],x[temp2,1],color = colors[i])
    plt.show()
        
if __name__=="__main__":
    # kmeans_train()
    # sgd_kmeans_with_fake_label()      
    kmeans_train()
            
            
            
    