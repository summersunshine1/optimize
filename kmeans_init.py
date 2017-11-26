import numpy as np
from random import uniform
# import types

def dis(center,x):
    d = np.square(center-x)
    return np.sqrt(np.min(np.sum(d,1)))
    
def get_closest_center(center,x):
    d = np.square(center-x)
    return np.argmin(np.sum(d,1))

def kmeans_plus(data,k,weights=0):
    samples = np.shape(data)[0]
    features = np.shape(data)[1]
    index = np.random.randint(samples)
    centers = [data[index]]
    if type(weights)==type(1) and weights == 0:
        weights = np.ones(samples)
    while len(centers)<k:
        cursum = 0
        diss = []
        for i in range(samples):
            dist = weights[i]*dis(centers,data[i,:])
            diss.append(dist)
            cursum += dist
        randdist = uniform(0,cursum)
        for i in range(len(diss)):
            randdist-=diss[i]
            if randdist<=0:
                centers.append(data[i])
                break
    centers = np.array(centers)
    return centers
           
def kmeans_plus_improve(data,k,l=2):
    samples = np.shape(data)[0]
    features = np.shape(data)[1]
    index = np.random.randint(samples)
    centers = [data[index]]
    isCenters = np.zeros(samples)
    # for i in range(samples):
        # dist+=dis(centers,data[i,:])
    initializationSteps = 5
    for i in range(initializationSteps):
        cursum = 0
        diss = []
        for j in range(samples):
            dist = dis(centers,data[j,:])
            diss.append(dist)
            cursum += dist
        for j in range(samples):
            p = l*diss[j]/cursum
            rand = uniform(0,1)
            if rand<p:
                if isCenters[j]:
                    continue
                isCenters[j] = 1
                centers.append(data[j,:])
    clusters = np.zeros(len(centers))
    for i in range(samples):
        c = get_closest_center(centers,data[i,:])
        clusters[c]+=1

    if len(clusters)>k:
        centers = np.array([t for t in centers])
        centers = kmeans_plus(centers,k,clusters)
    return centers
        
        
        
            
            
        
        
        
    
    