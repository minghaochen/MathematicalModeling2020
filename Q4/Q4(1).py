# -*- coding: utf-8 -*-
"""
Created on Sun Sep 20 12:05:55 2020

@author: jadew
"""
import pandas as pd
import numpy as np
import os
import math
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
from scipy import signal
from sklearn.decomposition import PCA
from numpy import fliplr
import random 


Train_data=np.zeros([0,4])
Test_data=np.zeros([0,4])
Train_labels=np.zeros([0])
Test_labels=np.zeros([0])

def takesometrain(per,D):
    train_num=math.floor(D.shape[0]*per)
    A=np.arange(D.shape[0])
    
    nums = random.sample(range(0,D.shape[0]),train_num)
    test_num=list(set(A)-set(nums))
    
    train_da=D[nums,1:5]

    train_labels=D[nums,0]-2
    
    test_da=D[test_num,1:5]
    
    test_labels=D[test_num,0]-2
    return train_da,train_labels,test_da,test_labels

# 训练数据的百分比
per=0.1
path='J:/BaiduNetdiskDownload/2020年中国研究生数学建模竞赛赛题/2020年C题/sleep.xlsx'
for n in range(5):
    D = pd.read_excel(path, sheet_name = n, header=None)
    D = D.values
    train_da,train_labels,test_da,test_labels=takesometrain(per,D)
    Train_data=np.vstack((Train_data,train_da))
    Test_data=np.vstack((Test_data,test_da))
    Train_labels=np.concatenate((Train_labels,train_labels))
    Test_labels=np.concatenate((Test_labels,test_labels))
    

# 增加第4维特征——能量和
fea4=Train_data.sum(axis=1)
Train_data4=np.c_[Train_data,fea4]

fea4t=Test_data.sum(axis=1)
Test_data4=np.c_[Test_data,fea4t]

# 增加第5维特征——theta/beta
fea5=Train_data[:,2]/Train_data[:,1]
Train_data5=np.c_[Train_data4,fea5]

fea5t=Test_data[:,2]/Test_data[:,1]
Test_data5=np.c_[Test_data4,fea5t]

# 增加第6维特征——beta/alpha
fea6=Train_data[:,1]/Train_data[:,0]
Train_data6=np.c_[Train_data5,fea6]

fea6t=Test_data[:,1]/Test_data[:,0]
Test_data6=np.c_[Test_data5,fea6t]

# 增加第7维特征——（theta + alpha)/beta
fea7=(Train_data[:,2]+Train_data[:,0])/Train_data[:,1]
Train_data7=np.c_[Train_data6,fea7]

fea7t=(Test_data[:,2]+Test_data[:,0])/Test_data[:,1]
Test_data7=np.c_[Test_data6,fea7t]

# 增加第8维特征——（theta + alpha)/（beta+alpha)
fea8=(Train_data[:,2]+Train_data[:,0])/(Train_data[:,1]+Train_data[:,0])
Train_data8=np.c_[Train_data7,fea8]

fea8t=(Test_data[:,2]+Test_data[:,0])/(Test_data[:,1]+Test_data[:,0])
Test_data8=np.c_[Test_data7,fea8t]


#PCA去除共线性——没啥用
#pca = PCA(n_components=8)
#pca.fit(Train_data8)
#print(pca.explained_variance_ratio_)
#Train_data8n = pca.transform(Train_data8)


# 使用的分类器
names = ["Nearest Neighbors", 
         "Linear SVM", 
         "RBF SVM", 
         "Gaussian Process",
        "Decision Tree", 
         "Random Forest", 
        # "Neural Net", 
         #"AdaBoost",
         #"Naive Bayes", 
         "QDA"
         ]

classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=1),
    SVC(gamma=0.1, C=10),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=7),
    RandomForestClassifier(max_depth=7, n_estimators=100, max_features=9),
   # MLPClassifier(alpha=0.35, max_iter=1000),
   # AdaBoostClassifier(),
   # GaussianNB(),
    QuadraticDiscriminantAnalysis()
    ]



# 训练
def classifiersTrain(DATA,LABEL):    
    CLFS = []
    for name, clf in zip(names, classifiers):
        clf.fit(DATA, LABEL)
        CLFS.append(clf)
    return CLFS
        
# 验证
def classfiersPredict(CLFS,DATA,LABEL):
    predictMatrix = np.zeros((len(CLFS),Test_data.shape[0]))
    for i in range(len(CLFS)):
        score = CLFS[i].score(DATA, LABEL)
        print(names[i],score)
        predictMatrix[i] = CLFS[i].predict(DATA)
    return predictMatrix
        
clfs = classifiersTrain(Train_data8, Train_labels)

PRE = classfiersPredict(clfs,Test_data8,Test_labels)








