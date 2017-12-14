import numpy as np
import pandas as pd
import json
import pickle
import os
import sklearn.preprocessing as pp
from scipy.sparse import csc_matrix
from sklearn import preprocessing
from sklearn.externals import joblib
import threading
import time
from sklearn import metrics
from getPath import *
pardir = getparentdir()

def write_dic(dic,path):
    with open(path,'wb') as f:
        # json.dump(dic, f)
        pickle.dump(dic, f)
    
def read_dic(path):
    with open(path,'rb') as f:
        dic = pickle.load(f)
    return dic
    
def test():
    dic = {1:2,3:4}
    write_dic(dic,'1.txt')
    dic1 = dict(read_dic('1.txt'))
    print(dic1==dic)
    
def write_record(df,path):
    if os.path.exists(path):
        df.to_csv(path,mode = 'a',encoding = 'utf-8',index = False,header = False)
    else:
        df.to_csv(path,mode = 'w',encoding='utf-8',index = False)
        
def compute_cos(a,b):
    a = csc_matrix(a)
    b = csc_matrix(b)
    vec_a = pp.normalize(a, axis=1)
    vec_b = pp.normalize(b, axis=1)
    res = vec_a*vec_b.T
    res = res.todense()
    return res.item(0)
    
def listfiles(rootDir): 
    list_dirs = os.walk(rootDir) 
    filepath_list = []
    for root, dirs, files in list_dirs:
        for f in files:
            filepath_list.append(os.path.join(root,f))
    return filepath_list
    
def start_thread(thread):
    thread.start()
    thread.join()
    
def cal_auc(predicted_ctr, labels):
    i_sorted = sorted(range(len(predicted_ctr)),key=lambda i: predicted_ctr[i],reverse=True)
    # print(predicted_ctr[i_sorted[:10]])
    tp = 0
    fp = 0
    last_tp = 0
    last_fp = 0
    lastscore = predicted_ctr[i_sorted[0]]+1
    x = []
    y = []
    auctemp = 0
    for i in range(len(predicted_ctr)):
        if lastscore!=predicted_ctr[i_sorted[i]]:
            auctemp += (fp-last_fp)*(tp+last_tp)/2
            last_tp = tp
            last_fp = fp
            lastscore = predicted_ctr[i_sorted[i]]
        if labels[i_sorted[i]]==1:
            tp+=1
        else:
            fp+=1
    auctemp+=(fp-last_fp)*(tp+last_tp)/2
    auctemp = auctemp/(fp*tp)
    return auctemp
    
def auc(pred,labels):
    fpr, tpr, thresholds = metrics.roc_curve(labels, pred, pos_label=1)
    return metrics.auc(fpr, tpr)
    

    
def write_middle_res(line,path):
    with open(path,'a',encoding='utf-8') as f:
        f.writelines(line)
        
def get_array_from_dic(dict,type):
    values = np.fromiter(iter(dicvalue), dtype=type)
    return values

if __name__=="__main__":
    a = [1,1,3,4,5]
    # b = [0,1,2,4]
    # print(compute_cos(a,b))
    # arr = ['a','b','c','c']
    # labels = convertLabels(arr,"1")
    # print(getlabels_detail(labels,'1'))
    # compare_res(pardir+'/data/res/rf_max_divde10_convert_feature_add_ll_3.csv',pardir+'/data/res/rf_max_divde10_convert_feature_add_ll_5.csv')
    # compare_res(pardir+'/data/res/rf_100_change_label_remove.csv',pardir+'/data/res/rf_max.csv')
    # remove_replicate_res(pardir+'/data/res/rfnew.csv')
    # getaccuracy(pardir+'/data/modeloutput')
    
    

