# -*- coding: utf-8 -*-
"""
Created on Fri Sep 18 20:41:03 2020

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

# 定义滤波器
def filterSignal(data):
    # 带通滤波 0.1 - 20 hz
    b, a = signal.butter(5, [2*0.1/250,2*10/250], 'bandpass')   #配置滤波器 8 表示滤波器的阶数
#    b, a = signal.butter(8, 2*20/250, 'lowpass')   
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

# 将正负样本从list转为nparray
trueAver = np.zeros((600,130,20))
falseAver = np.zeros((600,130,20))
for i in range(600):
    trueAver[i,0:130,0:20] = trueMatrix[i]
    falseAver[i,0:130,0:20] = falseMatrix[i]
# 全部数据
DataTotal = np.concatenate((trueAver,falseAver),axis=0)
LabelTotal = np.concatenate((np.ones(600),-1*np.ones(600)))
# 根据全部数据划分训练、验证
X_train, X_test, y_train, y_test = train_test_split(DataTotal, LabelTotal, test_size=.2, random_state=42, stratify = LabelTotal)    
    

# 训练数据中的正负样本矩阵
trueMatrix,falseMatrix = averageData(X_train,y_train)
# 产生2000训练数据
trainNum = 1000
trueAver = np.zeros((trainNum,130,20)) # 记录平均后的数据
falseAver = np.zeros((trainNum,130,20))
for i in range(trainNum):
    averNum = 5
    random.seed(888+i)
    nums = random.sample(range(0,480),averNum)
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


# 测试数据中的正负样本矩阵
trueMatrix,falseMatrix = averageData(X_test,y_test)
testNum = 120
trueAver = np.zeros((testNum,130,20))
falseAver = np.zeros((testNum,130,20))
for i in range(testNum):
    averNum = 5
    random.seed(666+i)
    nums = random.sample(range(0,testNum),averNum)
    mean1 = 0
    mean0 = 0
    for j in nums:
        mean1 += trueMatrix[j]
        mean0 += falseMatrix[j]
    trueAver[i,0:130,0:20] = mean1/averNum
    falseAver[i,0:130,0:20] = mean0/averNum
# 最终的测试数据    
DataTest = np.concatenate((trueAver,falseAver),axis=0)
LabelTest = np.concatenate((np.ones(testNum),-1*np.ones(testNum)))




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
    # 标准化数据
    if sc == None:      
        sc = StandardScaler()
        result = sc.fit_transform(Temp)
        return result,sc 
    else:
        result = sc.transform(Temp)
        return result
#    result = StandardScaler().fit_transform(Temp)
#    result = MinMaxScaler().fit_transform(Temp)
#    result = Temp
#    return result


# 全部数据放进去滤波和下采样和标准化再分离
#X = np.concatenate((DataTrain,DataTest),axis=0)
#transfromedMatrix = dataTransfrom(X,2*trainNum+int(1200*.2))
#transfromedMatrix_Train = transfromedMatrix[0:2*trainNum,:]
#transfromedMatrix_Test = transfromedMatrix[2*trainNum:2*trainNum+240,:]

transfromedMatrix_Train,sc = dataTransfrom(DataTrain,2000)
transfromedMatrix_Test = dataTransfrom(DataTest,240,sc)



# 使用的分类器
names = ["Nearest Neighbors", 
         "Linear SVM", 
         "RBF SVM", 
         "Gaussian Process",
         "Decision Tree", 
         "Random Forest", 
         "Neural Net", 
         "AdaBoost",
         "Naive Bayes", 
         "QDA",
         "LR"
         ]

classifiers = [
    KNeighborsClassifier(20),
    SVC(kernel="linear", C=0.025),
    SVC(gamma=0.1, C=1),
    GaussianProcessClassifier(1.0 * RBF(10)),
    DecisionTreeClassifier(max_depth=9),
    RandomForestClassifier(max_depth=9, n_estimators=100, max_features=50),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(n_estimators=100),
    GaussianNB(),
    QuadraticDiscriminantAnalysis(reg_param=0.1),
    LogisticRegression()
    ]



# 训练
def classifiersTrain(DATA,LABEL):    
    CLFS = []
    for name, clf in zip(names, classifiers):
        clf.fit(DATA, LABEL)
        CLFS.append(clf)
    return CLFS
        
# 验证
#def classfiersPredict(CLFS,DATA,LABEL):
#    predictMatrix = np.zeros((len(CLFS),240))
#    for i in range(len(CLFS)):
#        score = CLFS[i].score(DATA, LABEL)
#        print(names[i],score)
#        predictMatrix[i] = CLFS[i].predict(DATA)
#    return predictMatrix
        
clfs = classifiersTrain(transfromedMatrix_Train, LabelTrain)


# 某一个分类器的结果
#yread = clfs[6].predict(transfromedMatrix_Test)
#LabelTest
#f1_score(y_true, y_pred, *, labels=None, pos_label=1, average='binary', sample_weight=None, zero_division='warn')
#sklearn.metrics.roc_auc_score(y_true, y_score

def classfiersPredict(CLFS,DATA,LABEL):
    predictMatrix = np.zeros((len(CLFS),240))
    for i in range(len(CLFS)):
#        score = CLFS[i].score(DATA, LABEL)
        predictMatrix[i] = CLFS[i].predict(DATA)
        score1 = f1_score(LABEL,predictMatrix[i])
        score2 = roc_auc_score(LABEL,predictMatrix[i])
        print(names[i],'f1_score',score1,'roc_auc_score',score2)
        print('\n')
    return predictMatrix


PRE = classfiersPredict(clfs,transfromedMatrix_Test,LabelTest)



Result = []
Weight = [0.85,0.8291,0.6583,0.8458,0.8458,
          0.8791,0.9458,0.8875,0, 0.8333, 0.8416]
def testPerformance(Data,CLFS,Weight,randomselect = 0):
    for i in range(5):
        if i == 2 or i == 3:            
            for j in range(9):
                if randomselect == 1:                   
                    result = 0
                else:
                    dataAverage = np.zeros(12,130,20) 
                    for l in range(12):
                        temp = 0
                        for k in range(5):
                            temp += Data[i,j,k,l,0:130,0:20]
                        dataAverage[l,0:130,0:20] = temp/5
                    # 
                    testMatrix = dataTransfrom(dataAverage,12) # 返回 12*300
                    result = np.zeros(len(CLFS))
                    for k in range(len(CLFS)):               
                        result += Weight[k]*CLFS[k].predict(testMatrix)
                    predIndex = heapq.nlargest(2, range(len(result)),result.take)
                    Result.append(predIndex)            
        else:
            for j in range(10):
                if randomselect == 1:
                    result = 0                                      
                else:
                    dataAverage = np.zeros(12,130,20) 
                    for l in range(12):
                        temp = 0
                        for k in range(5):
                            temp += Data[i,j,k,l,0:130,0:20]
                        dataAverage[l,0:130,0:20] = temp/5
                    # 
                    testMatrix = dataTransfrom(dataAverage,12) # 返回 12*300
                    result = np.zeros(len(CLFS))
                    for k in range(len(CLFS)):               
                        result += Weight[k]*CLFS[k].predict(testMatrix)
                    predIndex = heapq.nlargest(2, range(len(result)),result.take)
                    Result.append(predIndex)
    return Result


            

np.save('transfromedMatrix_Train.npy',transfromedMatrix_Train)
np.save('LabelTrain.npy',LabelTrain)
np.save('transfromedMatrix_Test.npy',transfromedMatrix_Test)
np.save('LabelTest.npy',LabelTest)
