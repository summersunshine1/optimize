import numpy as np
from sklearn import linear_model
from sklearn import datasets
from sklearn.model_selection import train_test_split

class AdaDelta:
    def __init__(self,features,samples = 1):
        self.p = 0.96
        self.e=1e-10
        m = np.matrix([[0]]*features)
        self.lastg= m
        self.lasts =m
        self.samples = samples
        
        
    def getgrad(self,grad,iter):
        gradc = grad#/self.samples
        self.lastg = np.multiply(1-self.p,np.square(gradc))+np.multiply(self.p,self.lastg)
        delta = np.multiply(np.sqrt(self.lasts+self.e)/np.sqrt(self.lastg+self.e),gradc)
        self.lasts = np.multiply((1-self.p),np.square(delta))+np.multiply(self.p,self.lasts)
        return delta
        
class Adam:
    def __init__(self,paramnum,perparamnum = 1,alpha = 0.001,beta1 = 0.9,beta2 = 0.999,epsilo = 1e-8):
        self.alpha = alpha
        self.beta1 = beta1
        self.beta2 = beta2
        self.e = epsilo
        self.m = np.matrix([[0]*perparamnum]*paramnum)
        self.v = np.matrix([[0]*perparamnum]*paramnum)
    
    def getgrad(self,grad,iter):
        self.m = self.beta1*self.m+(1-self.beta1)*grad
        self.v = self.beta2*self.v+(1-self.beta2)*np.square(grad)
        mhat = self.m/(1-np.power(self.beta1,iter))
        vhat = self.v/(1-np.power(self.beta2,iter))
        tempa = self.alpha#/(np.sqrt(iter)
        delta = tempa*mhat/(np.sqrt(vhat)+self.e)
        return delta
        
    def getmaxgrad(self,grad,iter):#adamax
        self.m = self.beta1*self.m+(1-self.beta1)*grad
        self.v = np.maximum(self.beta2*self.v,np.abs(grad))
        delta = self.alpha/(1-np.power(self.beta1,iter))*self.m/(self.v+self.e)
        return delta
    

    
    