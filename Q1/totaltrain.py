# -*- coding: utf-8 -*-
"""
Created on Sat Sep 19 14:07:38 2020

@author: mhchen
"""
## 全部数据训练

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
LabelTrain = np.concatenate((np.ones(trainNum),-1*np.ones(trainNum)))

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

# 使用的分类器
names = [
#        "Nearest Neighbors", 
#         "Linear SVM", 
#         "RBF SVM", 
         "Gaussian Process",
#         "Decision Tree", 
#         "Random Forest", 
         "Neural Net", 
#         "AdaBoost",
#         "Naive Bayes", 
#         "QDA",
#         "LR"
         ]

classifiers = [
#    KNeighborsClassifier(20),
#    SVC(kernel="linear", C=0.025),
#    SVC(gamma=0.1, C=1),
    GaussianProcessClassifier(1.0 * RBF(10)),
#    DecisionTreeClassifier(max_depth=9),
#    RandomForestClassifier(max_depth=9, n_estimators=100, max_features=50),
    MLPClassifier(alpha=1, max_iter=1000),
#    AdaBoostClassifier(n_estimators=100),
#    GaussianNB(),
#    QuadraticDiscriminantAnalysis(reg_param=0.1),
#    LogisticRegression()
    ]



# 训练
def classifiersTrain(DATA,LABEL):    
    CLFS = []
    for name, clf in zip(names, classifiers):
        clf.fit(DATA, LABEL)
        CLFS.append(clf)
    return CLFS

clfs = classifiersTrain(transfromedMatrix_Train, LabelTrain)

import pickle
output = open('problem1.pkl', 'wb')
input = open('problem1.pkl', 'rb')
s = pickle.dump([clfs,sc], output)
output.close()
clfs,sc = pickle.load(input)
input.close()

# 加载测试数据
DataTest=np.load('Test_data.npy')  # 3600
# 人数、
Result = []
Weight = [0.85,0.8291,0.6583,0.8458,0.8458,
          0.8791,0.9458,0.8875,0, 0.8333, 0.8416]
##Weight = [0.5,0.5]
#def testPerformance(Data,CLFS,Weight,randomselect = 0):
#    for i in range(5):
#        if i == 1 or i == 2:            
#            for j in range(9):
#                if randomselect == 1:                   
#                    result = 0
#                else:
#                    dataAverage = np.zeros((12,130,20)) 
#                    for l in range(12):
#                        temp = 0
#                        for k in range(5):
#                            temp += Data[i,j,k,l,0:130,0:20]
#                        dataAverage[l,0:130,0:20] = temp/5
#                    # 
#                    testMatrix = dataTransfrom(dataAverage,12,sc) # 返回 12*300
#                    result = np.zeros(12)
#                    for k in range(len(CLFS)):               
#                        result += Weight[k]*CLFS[k].predict(testMatrix)
#                    predIndex = heapq.nlargest(2, range(len(result)),result.take)
#                    Result.append(predIndex)            
#        else:
#            for j in range(10):
#                if randomselect == 1:
#                    result = 0                                      
#                else:
#                    dataAverage = np.zeros((12,130,20))
#                    for l in range(12):
#                        temp = 0
#                        for k in range(5):
#                            temp += Data[i,j,k,l,0:130,0:20]
#                        dataAverage[l,0:130,0:20] = temp/5
#                    # 
#                    testMatrix = dataTransfrom(dataAverage,12) # 返回 12*300
#                    result = np.zeros(12)
#                    for k in range(len(CLFS)):               
#                        result += Weight[k]*CLFS[k].predict(testMatrix)
#                    predIndex = heapq.nlargest(2, range(len(result)),result.take)
#                    Result.append(predIndex)
#    return Result

Weight = [0.0,1]
def testPerformance(Data,CLFS,Weight,sc,randomselect = 0):
    Result = []
    Test = []
    for m in range(9):        
        dataAverage = np.zeros((12,130,20))    
        temp = 0
        for j in range(12):
            temp = 0        
            for i in range(5):    
                nums1 = random.sample(range(0,5),1)
                nums2 = random.sample(range(0,5),1)
                temp += Data[nums1,m,nums2,j,0:130,0:20]
            dataAverage[j,0:130,0:20] = temp/5
        testMatrix = dataTransfrom(dataAverage,12,sc)
#        Test.append(testMatrix)
        result = np.zeros(12)
        for k in range(len(CLFS)):               
            result += Weight[k]*CLFS[k].predict(testMatrix)
#            predIndex = heapq.nlargest(2, range(len(result)),result.take)
        Result.append(result)   
    return Result


Result = testPerformance(DataTest,clfs,Weight,sc,0)
#result = np.zeros(12)
#
#result = clfs[1].predict(testMatrix)


#np.save('DataRealTest.npy',Test)
