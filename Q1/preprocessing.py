# -*- coding: utf-8 -*-
"""
Created on Thu Sep 17 22:44:39 2020

@author: mh_chen
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

train_data = pd.read_excel('S1/S1_train_data.xlsx', sheet_name = 4, header=None)
train_event = pd.read_excel('S1/S1_train_event.xlsx', sheet_name = 4, header=None)



Mapping = {
            101: {1,7}, 102:{1,8}, 103:{1,9}, 104:{1,10}, 105:{1,11}, 106:{1,12},
            107: {2,7}, 108:{2,8}, 109:{2,9}, 110:{2,10}, 111:{2,11}, 112:{2,12},
            113: {3,7}, 114:{3,8}, 115:{3,9}, 116:{3,10}, 117:{3,11}, 118:{3,12},
            119: {4,7}, 120:{4,8}, 121:{4,9}, 122:{4,10}, 123:{4,11}, 124:{4,12},
            125: {5,7}, 126:{5,8}, 127:{5,9}, 128:{5,10}, 129:{5,11}, 130:{5,12},
            131: {6,7}, 132:{6,8}, 133:{6,9}, 134:{6,10}, 135:{6,11}, 136:{6,12},
            }




def takeOneSample(data,label,start=50,end=180):
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
start = 50
end = 180

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


trueAver = np.zeros((10,130,20))
falseAver = np.zeros((10,130,20))
for i in range(10):
    nums = [0,1,2,3,4,5,6,7,8,9]
    nums.remove(i)
    mean1 = 0
    mean0 = 0
    for j in nums:
        mean1 += trueMatrix[j]
        mean0 += falseMatrix[j]
    trueAver[i,0:130,0:20] = mean1/9
    falseAver[i,0:130,0:20] = mean0/9
    

Data1 = np.concatenate((trueAver,falseAver),axis=0)
Label1 = np.concatenate((np.ones(10),-1*np.ones(10)))

#Data1=np.load('Train_data.npy')
#Label1=np.load('Train_label.npy')

for i in range(len(Label1)):
    if Label1[i] == 0:
        Label1[i] = -1
        
        
# 画图
import matplotlib.pyplot as plt

def signalPlot(dataMatrix):
    
    plot_num = 1
    for i in range(20):
        plt.subplot(5, 4, plot_num)
        plt.plot(dataMatrix[:,i])
        plt.xticks(())
        plt.yticks(())
        # plt.text(.99, .01, ('%.2fs' % (t1 - t0)).lstrip('0'),
        #          transform=plt.gca().transAxes, size=15,
        #          horizontalalignment='right')
        plot_num += 1
    plt.show()
           
for j in range(20):
    Data1[16][:,j] = filterSignal(Data1[16][:,j])
signalPlot(Data1[16])           
            
#Data1[11]
#aver = Data1[11].mean(axis=1)
#plt.plot(aver)


#plt.plot(filtedData)
#import pywt
#
#
#def getEnergy(originColumn,maxlevel=2):
#    wp = pywt.WaveletPacket(originColumn, wavelet='db1',mode='symmetric',maxlevel=maxlevel)
#    n = maxlevel
#    re = []  #第n层所有节点的分解系数
#    for i in [node.path for node in wp.get_level(n, 'freq')]:
#        re.append(wp[i].data)
#    #第n层能量特征
#    energy = []
#    for i in re:
#        energy.append(pow(np.linalg.norm(i,ord=None),2))
#    return energy


from scipy import signal
def filterSignal(data):
    b, a = signal.butter(8, [2*0.1/250,2*20/250], 'bandpass')   #配置滤波器 8 表示滤波器的阶数
    filtedData = signal.filtfilt(b, a, data)  #data为要过滤的信号
    return filtedData


#energy = getEnergy(Data1[0][:,8])
    
#turnNum = 4
#sampleNum = turnNum*12*12*5

turnNum = 5
sampleNum = turnNum*12
sampleNum = 20
maxlevel = 2
def dataTransfrom(originMatrix):
    # 将每个通道数据用小波分解降维
#    transfromedMatrix = np.zeros((sampleNum,2**maxlevel,20)) 
    decimateDta = np.zeros((sampleNum,15,20))

    for i in range(len(originMatrix)):
        for j in range(20):
#            transfromedMatrix[i][:,j] = getEnergy(originMatrix[i][:,j])
            originMatrix[i][:,j] = filterSignal(originMatrix[i][:,j])
            decimateDta[i][:,j] = signal.decimate(originMatrix[i][:,j],9)
    # 降采样
    
    
    Temp = np.reshape(decimateDta, (sampleNum, 15*20))
    result = np.zeros((sampleNum,15*20))
    # 将每个样本从矩阵形式转化为向量
#    result = np.zeros((sampleNum,2**maxlevel*20))
    result = StandardScaler().fit_transform(Temp)
    return result

transfromedMatrix = dataTransfrom(Data1)

#pca = PCA(n_components=20)
#pca.fit(transfromedMatrix)
#print(pca.explained_variance_ratio_)
#transfromedMatrix = pca.transform(transfromedMatrix) 

#h = .02  # step size in the mesh

names = ["Nearest Neighbors", "Linear SVM", "RBF SVM", "Gaussian Process",
         "Decision Tree", "Random Forest", "Neural Net", "AdaBoost",
         "Naive Bayes", "QDA"]
class_weight = {1:10, -1:1}
classifiers = [
    KNeighborsClassifier(3),
    SVC(kernel="linear", C=0.001,class_weight={1: 10}),
    SVC(gamma=0.1, C=1,class_weight={1: 10}),
    GaussianProcessClassifier(1.0 * RBF(1.0)),
    DecisionTreeClassifier(max_depth=5),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=5),
    MLPClassifier(alpha=1, max_iter=1000),
    AdaBoostClassifier(),
    GaussianNB(),
    QuadraticDiscriminantAnalysis()]

def classifiersTrain(DATA,LABEL):    
    X_train, X_test, y_train, y_test = train_test_split(DATA, LABEL, test_size=.2, random_state=42, stratify = LABEL)
    CLFS = []
    for name, clf in zip(names, classifiers):
        clf.fit(X_train, y_train)
        score = clf.score(X_test, y_test)
#        clf.fit(DATA, LABEL)
#        score = clf.score(DATA, LABEL)
        print(name,score)
        CLFS.append(clf)
    return CLFS
        

def classfiersPredict(CLFS,DATA):
    predictMatrix = np.zeros((len(CLFS),4))
    for i in range(len(CLFS)):
        predictMatrix[i] = CLFS[i].predict(DATA)
    return predictMatrix
        
clfs = classifiersTrain(transfromedMatrix, Label1)

X_train, X_test, y_train, y_test = train_test_split(transfromedMatrix, Label1, test_size=.2, random_state=42, stratify = Label1)
PRE = classfiersPredict(clfs,X_test)
#PRE = PRE.mean(axis=0)




#
import lightgbm as lgb

trn_data = lgb.Dataset(transfromedMatrix_Train, LabelTrain)
val_data = lgb.Dataset(transfromedMatrix_Test, LabelTest)



params = {'num_leaves': 60, #结果对最终效果影响较大，越大值越好，太大会出现过拟合
          'min_data_in_leaf': 30,
          'is_unbalance' : True,
          'objective': 'binary', #定义的目标函数
          'max_depth': 7,
          'learning_rate': 0.01,
          "min_sum_hessian_in_leaf": 6,
          "boosting": "gbdt",
          "feature_fraction": 0.9,  #提取的特征比率
          "bagging_freq": 1,
          "bagging_fraction": 0.8,
          "bagging_seed": 11,
          "lambda_l1": 0.1,             #l1正则
          'lambda_l2': 0.1,     #l2正则
          "verbosity": -1,
          "nthread": -1,                #线程数量，-1表示全部线程，线程越多，运行的速度越快
          'metric': {'log_loss', 'auc'},  ##评价函数选择
          "random_state": 2019, #随机数种子，可以防止每次运行的结果不一致
          # 'device': 'gpu' ##如果安装的事gpu版本的lightgbm,可以加快运算
          }

clf = lgb.train(params,
                trn_data,
                1000,
                valid_sets=[trn_data,val_data],
                verbose_eval=20,
                early_stopping_rounds=60)
