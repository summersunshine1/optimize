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
    
def compare_res(res1,res2):
    d1 = pd.read_csv(res1)
    rowids = d1['row_id']
    shopids = d1['shop_id']
    d2 =pd.read_csv(res2)
    rowids1 = d2['row_id']
    shopids1 = d2['shop_id']
    length = 236798
    count=0
    dic = {}
    for i in range(length):
        if not rowids[i] in dic:
            dic[rowids[i]] = []
        dic[rowids[i]].append(shopids[i])
        if not rowids1[i] in dic:
            dic[rowids1[i]] = []
        dic[rowids1[i]].append(shopids1[i])
    
    for k,v in dic.items():
        # print(k)
        if len(v)<2:
            print(k)
            continue
        if v[0]==v[1]:
            count+=1
        
    print(count/length)
    
def convertLabels(arr,pickle_path):
    le = preprocessing.LabelEncoder()
    labels = le.fit_transform(arr)
    joblib.dump(le,pickle_path)
    return labels
    
def getlabels_detail(labels,pickle_path):
    le = joblib.load(pickle_path)
    arr = le.inverse_transform(labels)
    return arr   
    
def get_labels(pickle_path,arr):
    clf = joblib.load(pickle_path)
    labels = le.transform(arr)
    return labels
    
def process_wifi_info(wifi_info):
    f1 = lambda x : x.split('|')[0]
    f2 = lambda x : int(x.split('|')[1])
    f3 = lambda x : x.split('|')[2]
    arr = np.array(wifi_info.split(";"))
    bssids = list(np.fromiter((f1(xi) for xi in arr), arr.dtype, count=len(arr)))
    strengths= list(np.fromiter((f2(xi) for xi in arr), arr.dtype, count=len(arr)))
    connects= list(np.fromiter((f3(xi) for xi in arr), arr.dtype, count=len(arr)))
    return bssids,strengths,connects
    
def get_mallid_from_mallpath(path):
    mallpath = os.path.basename(path)
    mall_id = mallpath[:-4]
    return mall_id
    
def convert_str_to_bool(str):
    if str=="true":
        return True
    else:
        return False
    
def write_res_to_file(dic,respath):
    with open(respath,mode = 'w',encoding='utf-8') as f:
        f.writelines("row_id,shop_id\n")
        for item in dic:
            lines = str(item[0])+','+str(item[1])+'\n'
            f.writelines(lines)
            
          
def append_res_file(respath,evaluate_path):
    data = pd.read_csv(evaluate_path)
    row_ids = data['row_id']
    with open(respath,mode = 'a',encoding='utf-8') as f:
        for row_id in row_ids:
            lines = str(row_id)+',\n'
            f.writelines(lines)
            
def thread(func,argarr):
    processThread = threading.Thread(target=func, args=argarr) # <- 1 element list
    return processThread
    
def start_thread(thread):
    thread.start()
    thread.join()
    
def my_custom_loss_func(ground_truth, predictions):
    ground_truth = np.array(ground_truth)
    predictions = np.array(predictions)
    return len(ground_truth[ground_truth == predictions])/len(ground_truth)
    
def getlowmall(middlepath):
    path = middlepath
    malls = []
    with open(path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        arr = []
        for line in lines:
            a = float(line.split(':')[1])
            if a<0.8:
                malls.append(line.split(':')[0])
    return malls
    
def getaccuracy(middleoutput):
    path = middleoutput
    with open(path,'r',encoding='utf-8') as f:
        lines = f.readlines()
        arr = []
        for line in lines:
            a = float(line.split(':')[1])
            arr.append(a)
        m = np.mean(arr)
        print(m)
        
def remove_replicate_res(respath):
    with open(respath,'r',encoding='utf-8') as f:
        lines = f.readlines()
        kvdic = {}
        for line in lines[1:]:
            line = line.strip('\n')
            arr = line.split(',')
            kvdic[int(arr[0])]=arr[1]
    dict = sorted(kvdic.items(),key=lambda d:d[0])
    
    write_res_to_file(dict,respath)
    
def get_fix_date(dates):
    dates = pd.to_datetime(dates)
    length = len(dates)
    a_index= []
    b_index = []
    for i in range(length):
        if dates[i].day<25:
            a_index.append(i)
        else:
            b_index.append(i)
    return a_index,b_index

if __name__=="__main__":
    # a = [1,1,3,4,5]
    # b = [0,1,2,4]
    # print(compute_cos(a,b))
    # arr = ['a','b','c','c']
    # labels = convertLabels(arr,"1")
    # print(getlabels_detail(labels,'1'))
    compare_res(pardir+'/data/res/rf_max_divde10_convert_feature_add_ll_3.csv',pardir+'/data/res/rf_max_divde10_convert_feature_add_ll_5.csv')
    # compare_res(pardir+'/data/res/rf_100_change_label_remove.csv',pardir+'/data/res/rf_max.csv')
    # remove_replicate_res(pardir+'/data/res/rfnew.csv')
    # getaccuracy(pardir+'/data/modeloutput')
    
    

