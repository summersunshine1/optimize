import numpy as np

class AdaDelta:
    def __init__(self,gfunc):
        self.p = 0.95
        self.e=1e-6
        self.gfunc = gfunc
        self.g = [0]
        self.s = [0]
        
        
    def adadelta(iter,grad):
        temp = (1-p)*np.square(grad)+p*self.g[-1]
        self.g.append(temp)
        delta = -np.sqrt(s[-1]+self.e)/np.sqrt(temp+self.e)*grad
        t = (1-p)*np.square(delta)+p*self.s[-1]
        self.s.append(t)
        return delta
        
    
    