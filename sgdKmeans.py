import numpy as np
import random
from sklearn import datasets
import matplotlib.pylab as plt
from adadelta import *

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
    w = kmeans_plus(x,k)
    samples = np.shape(x)[0]
    features = np.shape(x)[1]
    clusters = np.array([-1]*samples)
    iters = 0
    print("batch kmeans begin train.......")
    while True:
        print("current iter"+str(iters))
        lastclusters = np.copy(clusters)
        for i in range(samples):
            dist = dis(w,x[i,:])
            tempk = np.argmin(np.sum(dist,1))
            clusters[i] = tempk
        for i in range(k):
            indexs = np.array(range(samples))
            kindexs = indexs[clusters==i]
            w[i,:] = np.mean(x[kindexs,:],0)
        if np.array_equal(clusters,lastclusters):
            break
        iters+=1   
    return w,clusters
    
def sgd_kmeans(x,k):
    w = initialize(x,k)
    maxiter = 1000
    samples = np.shape(x)[0]
    clusternum = np.array([0]*k)
    clusters = np.array([-1]*samples)
    print("sgd kmeans begin train.......")
    for iter in range(maxiter):
        print("current iter"+str(iter))
        indexs = list(range(samples))
        np.random.shuffle(indexs)
        lastclusters = np.copy(clusters)
        temp = 0
        for i in indexs:
            tempk = np.argmin(np.sum(dis(w,x[i,:]),1)) #get cluster nearest 
            clusternum[tempk]+=1 #nk=nk+1
            w[tempk,:]+=1/(clusternum[tempk])*(x[i,]-w[tempk,:])#w = w+1/n*(xi-wk)
            if np.array_equal(clusters,lastclusters):
                print("index"+str(temp))
                break
            temp+=1
    for i in range(samples):
        dist = dis(w,x[i,:])
        tempk = np.argmin(np.sum(dist,1))
        clusters[i] = tempk        
    return w,clusters
    
def kmeans_train():
    iris = datasets.load_iris()
    x = np.array(iris.data[:, [1,2]])
    y = np.array(iris.target)
    colors = ['b','g','r']
    # indexs = np.array(range(len(y)))
    # plt.subplot(131)    #nrows, ncols, plot_number
    # for i in range(3):
        # temp = indexs[y==i]
        # plt.scatter(x[temp,0],x[temp,1],color = colors[i])
    w,clusters = kmeans(x,3)
    # plt.subplot(132)
    # for i in range(3):
        # temp1 = indexs[clusters==i]
        # plt.scatter(x[temp1,0],x[temp1,1],color = colors[i])
    # w,clusters = sgd_kmeans(x,3)    
    # plt.subplot(133)
    # for i in range(3):
        # temp2 = indexs[clusters==i]
        # plt.scatter(x[temp2,0],x[temp2,1],color = colors[i])
    # plt.show()
        
if __name__=="__main__":
    kmeans_train()
            

            
            
            
    