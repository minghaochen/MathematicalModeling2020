# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 22:44:39 2020

@author: mh_chen
"""

import pandas as pd
import numpy as np
import os
path1='J:/BaiduNetdiskDownload/2020年中国研究生数学建模竞赛赛题/2020年C题/附件1-P300脑机接口数据'


startb = 50    #受到刺激后冲激在延时50-180个时刻内
endb = 180
Mapping = {
            101: {1,7}, 102:{1,8}, 103:{1,9}, 104:{1,10}, 105:{1,11}, 106:{1,12},
            107: {2,7}, 108:{2,8}, 109:{2,9}, 110:{2,10}, 111:{2,11}, 112:{2,12},
            113: {3,7}, 114:{3,8}, 115:{3,9}, 116:{3,10}, 117:{3,11}, 118:{3,12},
            119: {4,7}, 120:{4,8}, 121:{4,9}, 122:{4,10}, 123:{4,11}, 124:{4,12},
            125: {5,7}, 126:{5,8}, 127:{5,9}, 128:{5,10}, 129:{5,11}, 130:{5,12},
            131: {6,7}, 132:{6,8}, 133:{6,9}, 134:{6,10}, 135:{6,11}, 136:{6,12},
            }


def takeOneSample(data,label,startb,endb):
    DataMatrix = np.zeros((label.shape[0],endb-startb,data.shape[1])) # 60=12*5
    trueLabel = np.zeros(label.shape[0])
    for i in range(len(label)):
        if label[0][i] != 100:
            DataMatrix[i,0:endb-startb,0:20] = data[label[1][i]+startb-1:label[1][i]+endb-1]
        if label[0][i] in Mapping[label[0][0]]:
            trueLabel[i] = 1
    index = [0,13,26,39,52,65]
    trueLabel = np.delete(trueLabel, index)
    DataMatrix = np.delete(DataMatrix, index, axis=0)            
    return DataMatrix,trueLabel



def takeTestSample(data,label,startb,endb):
    DataMatrix = np.zeros((label.shape[0],endb-startb,data.shape[1])) # 60=12*5
    trueLabel = np.zeros(label.shape[0])
    for i in range(len(label)):
        if label[0][i] != 100:
            DataMatrix[i,0:endb-startb,0:20] = data[label[1][i]+startb-1:label[1][i]+endb-1]
    index = [0,13,26,39,52,65]
    trueLabel = np.delete(trueLabel, index)
    DataMatrix = np.delete(DataMatrix, index, axis=0)            
    return DataMatrix


def Getcycledata(cycle):
    Person=['S1','S2','S3','S4','S5']
    for p in range(len(Person)):         
        key=Person[p]
        path2='/'+key
        path_final=path1+path2
        for num in range(12):
            train_data = pd.read_excel(path_final+'/train_data.xlsx', sheet_name = num, header=None)
            train_event = pd.read_excel(path_final+'/train_event.xlsx', sheet_name = num, header=None)
            D,L= takeOneSample(train_data,train_event,startb,endb)
            D_cycle=D[0:cycle*12,:,:]
            L_cycle=L[0:cycle*12]    
            if num==0 and p ==0:
                Train_data=D_cycle
                Train_label=L_cycle
            else:
                Train_data=np.vstack((Train_data, D_cycle)) 
                Train_label=np.concatenate((Train_label, L_cycle),axis=0) 
    return Train_data,Train_label


Train_data,Train_label=Getcycledata(cycle=5)



def Gettestdata():
    Person=['S1','S2','S3','S4','S5']
    for p in range(len(Person)):         
        key=Person[p]
        path2='/'+key
        path_final=path1+path2
        if key == 'S2' or key =='S3':
            N=9
        else:
            N=10
        for num in range(N):
            test_data = pd.read_excel(path_final+'/test_data.xlsx', sheet_name = num, header=None)
            test_event = pd.read_excel(path_final+'/test_event.xlsx', sheet_name = num, header=None)
            D= takeTestSample(test_data,test_event,startb,endb)
            if num==0 and p ==0:
                Test_data=D
                
            else:
                Test_data=np.vstack((Test_data, D)) 
                
    return Test_data

Test_data=Gettestdata()

#  六维数组取测试数据


#这个函数的功能是切出某个字母中的某一轮，包括12次激励，把激励从1-12进行排序，对应的130*20的片段也根据从1-12行激励排序
def Increasing(D,E,cycle):
    DataMatrix = np.zeros((12,130,20)) # 60=12*5
    Event = np.zeros(12)
    Time=np.zeros(12)
    Event=E[13*cycle+1:13*cycle+13][0].values
    Time=E[13*cycle+1:13*cycle+13][1].values
    index=np.argsort(Event)
    Event_sorted=Event[index]
    Time_sorted=Time[index]
    for i in range(12):
        DataMatrix[i,0:130,0:20] = D[Time_sorted[i]+50-1:Time_sorted[i]+180-1]
    return DataMatrix, Event_sorted


path1='J:/BaiduNetdiskDownload/2020年中国研究生数学建模竞赛赛题/2020年C题/附件1-P300脑机接口数据'


Test_data=np.zeros((5,12,5,12,130,20),dtype=float)
Person=['S1','S2','S3','S4','S5']
for p in range(len(Person)):
    key=Person[p]
    path2='/'+key
    path_final=path1+path2
    if key == 'S2' or key =='S3':
        N=9
    else:
        N=10
    for n in range(N):
        D = pd.read_excel(path_final+'/test_data.xlsx', sheet_name = n, header=None)
        E = pd.read_excel(path_final+'/test_event.xlsx', sheet_name = n, header=None)      
        for c in range(5):
            DataMatrix, Event_sorted = Increasing(D,E,c)              
            for b in range(12):           
                Test_data[p,n,c,b,:]= DataMatrix[b,:,:]
                

np.save("J:/BaiduNetdiskDownload/2020年中国研究生数学建模竞赛赛题/2020年C题/附件1-P300脑机接口数据/Test_data.npy", Test_data)

##
np.save("J:/BaiduNetdiskDownload/2020年中国研究生数学建模竞赛赛题/2020年C题/附件1-P300脑机接口数据/5轮/Train_data.npy", Train_data) 
np.save("J:/BaiduNetdiskDownload/2020年中国研究生数学建模竞赛赛题/2020年C题/附件1-P300脑机接口数据/5轮/Train_label.npy", Train_label) 



        
