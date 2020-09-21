# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 07:45:02 2020

@author: mhchen
"""

import random 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis

from sklearn.decomposition import PCA

train_data = pd.read_excel('S5/S5_train_data.xlsx', sheet_name = 0, header=None)
train_event = pd.read_excel('S5/S5_train_event.xlsx', sheet_name = 0, header=None)



Mapping = {
            101: {1,7}, 102:{1,8}, 103:{1,9}, 104:{1,10}, 105:{1,11}, 106:{1,12},
            107: {2,7}, 108:{2,8}, 109:{2,9}, 110:{2,10}, 111:{2,11}, 112:{2,12},
            113: {3,7}, 114:{3,8}, 115:{3,9}, 116:{3,10}, 117:{3,11}, 118:{3,12},
            119: {4,7}, 120:{4,8}, 121:{4,9}, 122:{4,10}, 123:{4,11}, 124:{4,12},
            125: {5,7}, 126:{5,8}, 127:{5,9}, 128:{5,10}, 129:{5,11}, 130:{5,12},
            131: {6,7}, 132:{6,8}, 133:{6,9}, 134:{6,10}, 135:{6,11}, 136:{6,12},
            }




def takeOneSample(data,label,start=0,end=200):
    DataMatrix = np.zeros((66,end-start,20)) # 60=12*5
    trueLabel = np.zeros(66)
    for i in range(len(label)):
        if label[0][i] != 100:
            DataMatrix[i,0:end-start,0:20] = data[label[1][i]+start:label[1][i]+end]
        if label[0][i] in Mapping[label[0][0]]:
            trueLabel[i] = 1
    index = [0,13,26,39,52,65]
    trueLabel = np.delete(trueLabel, index)
    DataMatrix = np.delete(DataMatrix, index, axis=0)            
    return trueLabel,DataMatrix



# 采样周期40ms
start = 0
end = 200

Label1,Data1 = takeOneSample(train_data,train_event,start,end)

def averageData(data,label):
    trueMatrix = []
    falseMatrix = []
    for i in range(len(label)):
        if label[i] == 1:
#            np.append(trueMatrix,data[i])
            trueMatrix.append(data[i])
        else:
#            np.append(falseMatrix,data[i])
            falseMatrix.append(data[i])
    return trueMatrix,falseMatrix

trueMatrix,falseMatrix = averageData(Data1,Label1)


trueAver = np.zeros((10,200,20))
falseAver = np.zeros((10,200,20))
for i in range(10):
    nums = [0,1,2,3,4,5,6,7,8,9]
    nums.remove(i)
    mean1 = 0
    mean0 = 0
    for j in nums:
        mean1 += trueMatrix[j]
        mean0 += falseMatrix[j]
    trueAver[i,0:200,0:20] = mean1/9
    falseAver[i,0:200,0:20] = mean0/9
    

Data1 = np.concatenate((trueAver,falseAver),axis=0)
Label1 = np.concatenate((np.ones(10),0*np.ones(10)))

from scipy import signal
def filterSignal(data):
    b, a = signal.butter(8, 2*20/250, 'lowpass')   #配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data)  #data为要过滤的信号
    return filtedData

turnNum = 5
sampleNum = turnNum*12
sampleNum = 20
def dataTransfrom(originMatrix):
    # 将每个通道数据用小波分解降维
    decimateDta = np.zeros((sampleNum,20,20))

    for i in range(len(originMatrix)):
        for j in range(20):
            originMatrix[i][:,j] = filterSignal(originMatrix[i][:,j])
            decimateDta[i][:,j] = signal.decimate(originMatrix[i][:,j],10)
    # 降采样
    
    Temp = np.reshape(decimateDta, (sampleNum, 20*20))
    result = np.zeros((sampleNum,20*20))
    # 将每个样本从矩阵形式转化为向量
#    result = np.zeros((sampleNum,2**maxlevel*20))
    result = StandardScaler().fit_transform(Temp)
    return result

transfromedMatrix = dataTransfrom(Data1)


from group_lasso import LogisticGroupLasso

#np.random.seed(0)
LogisticGroupLasso.LOG_LOSSES = True


group_sizes = [np.ones(20) for i in range(1,21)]
groups = np.concatenate([size * [i] for i, size in enumerate(group_sizes)])
num_coeffs = sum(group_sizes)


gl = LogisticGroupLasso(
    groups=groups,
    group_reg=0.03,
    l1_reg=0.0,
    n_iter = 1000,
    scale_reg="inverse_group_size",
#    subsampling_scheme=1,
    supress_warning=True,
)

gl.fit(transfromedMatrix, Label1)

pred_c = gl.predict(transfromedMatrix)
sparsity_mask = gl.sparsity_mask_


# Compute performance metrics
accuracy = (pred_c == Label1).mean()

# Print results
print(f"Number variables: {len(sparsity_mask)}")
print(f"Number of chosen variables: {sparsity_mask.sum()}")




