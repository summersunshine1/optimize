import matplotlib.pylab as plt
import numpy as np
import math

from commonLib import *
from getPath import *
pardir = getparentdir()
resdir = pardir+'/res'

def read_data(path):
    acc = []
    auc = []
    loss = []
    with open(path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split()
            if len(arr)<3:
                continue
            for a in arr:
                barr = a.split(':')
                if len(barr)<2:
                    continue
                temp = float(barr[1])
                if math.isnan(temp):
                    continue
                if barr[0]=='acc':
                    acc.append(temp)
                elif barr[0]=='auc':
                    auc.append(temp)
                elif barr[0]=='loss': 
                    loss.append(temp)
    return acc,auc,loss
    
def plot_data():
    files = listfiles(resdir)
    for file in files:
        name = os.path.basename(file)
        acc,auc,loss = read_data(file)
        # plt.title(name)
        plt.plot(acc,label='acc')
        plt.plot(auc,label=name+' auc')
        plt.plot(loss,label=name+' loss')
        plt.legend()
        plt.show()
        
if __name__=="__main__":
    plot_data()
                    