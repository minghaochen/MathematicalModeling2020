# -*- coding: utf-8 -*-
"""
Created on Mon Sep 21 00:39:51 2020

@author: mhchen
"""

import random 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
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
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,roc_auc_score
import heapq 

Data1=np.load('Train_data.npy')  # 3600
Label1=np.load('Train_label.npy')
from group_lasso import LogisticGroupLasso

#np.random.seed(0)
LogisticGroupLasso.LOG_LOSSES = True
# 定义滤波器
def filterSignal(data):
    # 带通滤波 0.1 - 20 hz
    b, a = signal.butter(8, [2*0.23/250,2*30/250], 'bandpass')   #配置滤波器 8 表示滤波器的阶数
#    b, a = signal.butter(8, 2*60/250, 'lowpass')   
    filtedData = signal.filtfilt(b, a, data)  #data为要过滤的信号
    return filtedData

# 从数据中分离正负样本
def averageData(data,label):
    trueMatrix = []
    falseMatrix = []
    for i in range(len(label)):
        if label[i] == 1:
            trueMatrix.append(data[i])
        else:
            falseMatrix.append(data[i])
    return trueMatrix,falseMatrix

# 所有数据中的正负样本 
trueMatrix,falseMatrix = averageData(Data1,Label1)

# 产生2000训练数据
trainNum = 600
trueAver = np.zeros((trainNum,130,20)) # 记录平均后的数据
falseAver = np.zeros((trainNum,130,20))
for i in range(trainNum):
    averNum = 5
    random.seed(888+i)
    nums = random.sample(range(0,600),averNum)
    mean1 = 0
    mean0 = 0
    for j in nums:
        mean1 += trueMatrix[j]
        mean0 += falseMatrix[j]
    trueAver[i,0:130,0:20] = mean1/averNum
    falseAver[i,0:130,0:20] = mean0/averNum
# 最终的训练数据    
DataTrain = np.concatenate((trueAver,falseAver),axis=0)
LabelTrain = np.concatenate((np.ones(trainNum),0*np.ones(trainNum)))

# 将数据滤波和下采样
def dataTransfrom(originMatrix,sampleNum,sc=None):
    decimateDta = np.zeros((sampleNum,15,20))
    for i in range(len(originMatrix)):
        for j in range(20):
            originMatrix[i][:,j] = filterSignal(originMatrix[i][:,j])
            decimateDta[i][:,j] = signal.decimate(originMatrix[i][:,j],9)    
    Temp = np.reshape(decimateDta, (sampleNum, 15*20))
    result = np.zeros((sampleNum,15*20))
    # 标准化数据
    if sc == None:      
        sc = StandardScaler()
        result = sc.fit_transform(Temp)
        return result,sc 
    else:
        result = sc.transform(Temp)
        return result

transfromedMatrix_Train,sc = dataTransfrom(DataTrain,trainNum*2)

group_sizes = [np.ones(15) for i in range(1,21)]
groups = np.concatenate([size * [i] for i, size in enumerate(group_sizes)])
num_coeffs = sum(group_sizes)


gl = LogisticGroupLasso(
    groups=groups,
    group_reg=0.04,
    l1_reg=0.0,
    n_iter = 1000,
    scale_reg="inverse_group_size",
#    subsampling_scheme=1,
    supress_warning=True,
)

gl.fit(transfromedMatrix_Train, LabelTrain)

pred_c = gl.predict(transfromedMatrix_Train)
sparsity_mask = gl.sparsity_mask_


# Compute performance metrics
accuracy = (pred_c == LabelTrain).mean()

# Print results
print(f"Number variables: {len(sparsity_mask)}")
print(f"Number of chosen variables: {sparsity_mask.sum()}")

