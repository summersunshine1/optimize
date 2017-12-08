import numpy as np
import os
from getPath import *
pardir = getparentdir()
from commonLib import *

data_dir = pardir+'/data/'

def read_data(path,istrain):
    with open(path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        labels = []
        train_data = []
        for line in lines:
            arr = line.split()
            if int(arr[0])==-1:
                labels.append([0])
            else:
                labels.append([1])
            train = arr[1:]
            temp = np.array([0]*999996)
            for t in train:
                tarr = t.split(':')
                temp[int(tarr[1])-1]=1
            train_data.append(temp)
        train_data = np.array(train_data)
        labels = np.array(labels)
        data = np.hstack((train_data,labels))
        print(np.shape(data))
        if istrain:
            write_dic(data,data_dir+"trainffm")
        else:
            write_dic(data,data_dir+"testffm")
        
if __name__=="__main__":
    read_data(data_dir+'train.ffm',1)
    read_data(data_dir+'test.ffm',0)
    