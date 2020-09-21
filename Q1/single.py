# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 10:08:10 2020

@author: mhchen
"""


import random 
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler,MinMaxScaler
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC,LinearSVC
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import RBF
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.discriminant_analysis import QuadraticDiscriminantAnalysis,LinearDiscriminantAnalysis
from scipy import signal
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score,roc_auc_score
import heapq 

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
    rowcolumn = np.zeros(66)
    for i in range(len(label)):
        rowcolumn[i] =  label[0][i]
        if label[0][i] != 100:
            DataMatrix[i,0:end-start,0:20] = data[label[1][i]+start:label[1][i]+end]
        if label[0][i] in Mapping[label[0][0]]:
            trueLabel[i] = 1
    index = [0,13,26,39,52,65]
    trueLabel = np.delete(trueLabel, index)
    rowcolumn = np.delete(rowcolumn, index)
    DataMatrix = np.delete(DataMatrix, index, axis=0)            
    return trueLabel,DataMatrix,rowcolumn,Mapping[label[0][0]]


CLFS = []
SCORES = []
for character in range(12):
    
    train_data1 = pd.read_excel('S5/S5_train_data.xlsx', sheet_name = character, header=None)
    train_event1 = pd.read_excel('S5/S5_train_event.xlsx', sheet_name = character, header=None)
    
    # 通道平均
    Label1,Data1,Flash1,Char1 = takeOneSample(train_data1,train_event1)
    averData1 = np.zeros((60,200))
    for i in range(60):
        mean = 0
        for j in range(20):
            mean += Data1[i,:,j]
        averData1[i] = mean/20
    
    # 滑动滤波
    n = 11
    filtered = np.zeros((60,200-n+1))
    for j in range(60):
        filtered[j,:] = np.convolve(averData1[j,:], np.ones((n,))/n, mode="valid")
    
    # 下采样
    decimateData = np.zeros((60,19))
    for i in range(60):
        decimateData[i,:] = signal.decimate(filtered[i,:],10)  
    
    # 选择相同数量样本进行训练
    trueindex = np.argwhere(Label1==1)
    falseindex = np.argwhere(Label1==0)
    nums = random.sample(range(0,50),10)
    TrueData = decimateData[trueindex,:].reshape(10,19)
    FalseData = decimateData[falseindex[nums],:].reshape(10,19)
    # 最终的训练数据    
    DataTrain = np.concatenate((TrueData,FalseData),axis=0)
    LabelTrain = np.concatenate((np.ones(10),-1*np.ones(10)))
    
    classifiers = [
        SVC(kernel="linear", C=0.025),
#        LinearSVC()
#        SVC(gamma=10, C=1),
#        MLPClassifier(alpha=1, max_iter=1000),
#        QuadraticDiscriminantAnalysis(reg_param=0.1),
#        LinearDiscriminantAnalysis(solver='lsqr', shrinkage='auto'),
#        LogisticRegression()
        ]
    
    names = [
             "Linear SVM", 
#             "RBF SVM", 
#             "Neural Net", 
#             "QDA",
#             "LDA",
#             "LR"
             ]
    
    scores = []
    clfs = []
    for name, clf in zip(names, classifiers):
        clf.fit(DataTrain, LabelTrain)
        score = clf.score(DataTrain, LabelTrain)
        clfs.append(clf)
        scores.append(score)
        print(name,score)
    
    CLFS.append(clfs)
    SCORES.append(scores)
    


## 处理测试数据


def takeOneSampleTest(data,label,start=0,end=200):
    DataMatrix = np.zeros((66,200,20)) # 60=12*5
    rowcolumn = np.zeros(66)
    for i in range(len(label)):
        rowcolumn[i] =  label[0][i]
        DataMatrix[i,0:200,0:20] = data[label[1][i]:label[1][i]+200]
    index = [0,13,26,39,52,65]
    rowcolumn = np.delete(rowcolumn, index)
    DataMatrix = np.delete(DataMatrix, index, axis=0)            
    return DataMatrix,rowcolumn

for character in range(10):
    test_data1 = pd.read_excel('S5/S5_test_data.xlsx', sheet_name = character, header=None)
    test_event1 = pd.read_excel('S5/S5_test_event.xlsx', sheet_name = character, header=None)
    
    Data1,Flash1 = takeOneSampleTest(test_data1,test_event1)
    averData1 = np.zeros((60,200))
    for i in range(60):
        mean = 0
        for j in range(20):
            mean += Data1[i,:,j]
        averData1[i] = mean/20
    
    # 滑动滤波
    n = 11
    filtered = np.zeros((60,200-n+1))
    for j in range(60):
        filtered[j,:] = np.convolve(averData1[j,:], np.ones((n,))/n, mode="valid")
    
    # 下采样
    decimateData = np.zeros((60,19))
    for i in range(60):
        decimateData[i,:] = signal.decimate(filtered[i,:],10) 
    
#    ROW = []
#    COLUMN = []
    Result_row = 0
    Result_column =0
    for turn in range(5):
        row = []
        column = []
        for char in range(1,7):
            row.append(np.argwhere(Flash1==char)[turn])
            column.append(np.argwhere(Flash1==char+6)[turn])
#        ROW.append(row)
#        COLUMN.append(column)
        
        # 分别对行列测试
        TestData = decimateData[row,:].reshape(6,19)
        Pred_row = 0
        for clf in CLFS:
            Pred_row += clf[0].predict(TestData)   
        Result_row += Pred_row  
        
        TestData = decimateData[column,:].reshape(6,19)
        Pred_column = 0
        for clf in CLFS:
            Pred_column += clf[0].predict(TestData)
        Result_column += Pred_column
        
    print(Result_row)
    print(Result_column)
         
        
    
        
        
    
    


    

#
#
#n=11
#filtered = np.zeros((train_data1.shape[0],200-n+1,train_data1.shape[1]))
#for j in range(train_data1.shape[1]):
#    filtered[i,:,j] = np.convolve(data[i,:,j], np.ones((n,))/n,mode="same")
#
#train_data1 = pd.DataFrame(StandardScaler().fit_transform(filtered))
#
#
#decimateData = np.zeros((Data1.shape[0],20,Data1.shape[2]))
#for i in range(60):
#    for j in range(20):
#        decimateData[i,:,j] = signal.decimate(Data1[i,:,j],10)
#decimateData = decimateData.reshape(60,400)
#
## 画图
#import matplotlib.pyplot as plt
#
#def signalPlot(dataMatrix):
#    plot_num = 1
#    for i in range(20):
#        plt.subplot(5, 4, plot_num)
#        plt.plot(dataMatrix[:,i])
#        plt.xticks(())
#        plt.yticks(())
#        plot_num += 1
#    plt.show()
           




