import numpy as np
from getPath import *
pardir = getparentdir()
train_path = pardir+'/data/train.ffm'
test_path = pardir+'/data/test.ffm'

def analyze(path):
    labels = []
    with open(path,'r',encoding = 'utf-8') as f:
        lines = f.readlines()
        for line in lines:
            arr = line.split()
            labels.append(int(arr[0]))
    labels = np.array(labels)
    posratio = len(labels[labels==1])/len(labels)
    print(posratio)
    
if __name__=="__main__":
    analyze(train_path)
    analyze(test_path)
    