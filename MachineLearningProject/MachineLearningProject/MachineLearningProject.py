
# -*- coding: utf-8 -*-
"""
Created on Fri Dec  8 17:46:44 2017
@author: onur.gultekin
"""
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from mpl_toolkits.mplot3d import Axes3D
from sklearn import preprocessing
from sklearn.decomposition import PCA
from sklearn.cluster import(KMeans, DBSCAN)
from sklearn import neighbors
from sklearn.model_selection import train_test_split
import datetime as dt

def prepareData():
    #get data from excel
    excelData = pd.read_excel("data.xls", sheet_name="Games", skiprows=1)
    excelData = excelData.loc[:, 'Zaman':'Ödül']
    

    #create string to numeric mapping key-value pairs for each string column
    teamMapping = {'BT': 0, 'AL': 1, 'AB': 2, 'BL': 3, 'AT':4, 'LT' : 5}
    playerMapping = {'A': 1, 'L': 2, 'B': 3, 'T': 4}
    zoneMapping = {'Yok': 0, 'Var': 1, 'Sür': 2, 'Hayır': 0, 'Evet': 1}
    kontrMapping = {'Yok': 1, 'Sür': 2, 'Var': 3}
    colorMapping = {'Kozsuz': 1, 'Kupa': 2, 'Maça': 3, 'Karo': 4, 'Sinek': 5}
    targetMapping = {'Partial': 1, 'Zon': 2}

    #replace string columns with numeric values
    data = pd.DataFrame.copy(excelData)
    data["Takım"] = excelData["Takım"].replace(teamMapping)
    data["Oyuncu"] = excelData["Oyuncu"].replace(playerMapping)
    data["Zon"] = excelData["Zon"].replace(zoneMapping)
    data["Renk"] = excelData["Renk"].replace(colorMapping)
    data["Kontr"] = excelData["Kontr"].replace(kontrMapping)
    data["Hedef"] = excelData["Hedef"].replace(targetMapping)
    
    #convert time values to numeric data by a created formula
    times = []
    for i, row in excelData.iterrows():
       time = (row["Zaman"].hour - 8) * 100 + row["Zaman"].minute
       times.append(time)
    data["Zaman"] = times
    
    print("Processing {} samples with {} attributes".format(len(data.index), len(data.columns)))
    return pd.DataFrame(data), excelData

def scaleData(X):
    scaler = StandardScaler()
    X = scaler.fit_transform(X)
    return X

def doPCA(X, k):
    #reduce the dimensionality of data by PCA that identifies the components of
    #the data with largest variance
    #and transform data according to new components
    pca = PCA(n_components = k)
    X = pca.fit_transform(X)
    print()
    #print cumulative eigen value ratios
    print("cumulative eigen value ratios")
    print(pca.explained_variance_ratio_.cumsum())
    print()
    #print eigen vectors.
    for i in range(pca.components_.shape[0]):
        print("eigen vector " + str(i + 1))
        print(pca.components_[i])
    return X

def plot1dData(X):
    #plot 1-dimensional data
    plt.plot(X, np.zeros_like(X), 'x')
    
def onPick(event):
    ind = event.ind[0]
    values = excelData.values[ind]
    if len(values)>0:
        print()
        print("Zaman Takim Oyncu Puan Dagılım Zon Kontrat Renk Kontr Hedef Sonuç Ödül")
        print(values)

def plot2dData(X):
    fig = plt.figure()
    ax = plt.axes()
    ax.plot(X[:,0], X[:,1], 'o', markersize=7, color='blue', alpha=0.6, picker = True)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Data points on 2-dimension after PCA')
    fig.canvas.mpl_connect('pick_event', onPick)
    plt.show()

def plot3dData(X):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    x = X[:,0]
    y = X[:,1]
    z = X[:,2]
    ax.scatter(x, y, -z, zdir='z', c= 'red')
    plt.show()

def plot2dDataByColoredFeatures(X):
    fig = plt.figure()
    ax = plt.axes()

    #plot data by colored features
    for i in range(X.shape[0]):
        ax.scatter(X[i,0], X[i,1], color = getColor(data.values[i,4], data.values[i,8]), alpha=0.6)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Data points on 2-dimension after PCA')
    plt.show()

def scaleAndTransformByTarget(X):
    X = preprocessing.scale(data)
    pca = PCA(n_components = 2)
    X = pca.fit_transform(X)
    print(pca.explained_variance_)
    colors = {1:'red', 2:'blue'}
    plt.scatter(X[:,0], X[:,1], c=data["Hedef"].replace(colors),alpha=0.6)
    targetPartial = mpatches.Patch(color=colors[1], label='Hedef = Partial')
    targetZone = mpatches.Patch(color=colors[2], label='Hedef = Zone')
    plt.legend(handles=[targetZone, targetPartial])
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    plt.title('Normalized data points on 2-dimension after PCA')
    plt.show()

def choose_k_dimension(data):
    pca = PCA(n_components = len(data[1]))
    X = pca.fit(data)
    
    k = 0
    variance_ratio = 0
    for i in range(len(pca.explained_variance_ratio_)):
        variance_ratio += pca.explained_variance_ratio_[i]
        if variance_ratio > 0.9:
            k = i + 1
            break
    
    print("found dimension k: {} with PoV: {}".format(k, variance_ratio))
    #visualize variance to dimension graph
    klist = range(1, pca.n_components + 1)
    plt.plot(klist[:], pca.explained_variance_ratio_.cumsum() * 100)
    plt.plot(k, variance_ratio * 100, 'X',markersize=12, color='red')
    plt.xlabel('k = Dimension Count')
    plt.ylabel('PoV = Proportion of Variance')
    plt.title('Choosing k dimension by Proportion of Variance')
    plt.xticks(klist)
    plt.show()
    return k

def getColor(x, y):
    zoneMapping = {'Yok': 0, 'Var': 1, 'Hayır': 0, 'Evet': 1}
    targetMapping = {'Partial': 1, 'Zon': 2}
    if x == 0 and y == 1: #zon yok, hedef partial
        return 'green'
    elif x == 0 and y == 2: #zon yok, hedef zon
        return 'red'
    elif x == 1 and y == 1: #zon var, hedef partial
        return "purple"
    elif x == 1 and y == 2: #zon var, hedef zon
       return "blue"

def cluster_with_DBScan(X):
    db = DBSCAN(eps=0.8, min_samples=15).fit(X)
    core_samples_mask = np.zeros_like(db.labels_, dtype=bool)
    core_samples_mask[db.core_sample_indices_] = True
    labels = db.labels_
    # Number of clusters in labels, ignoring noise if present.
    n_clusters_ = len(set(labels)) - (1 if -1 in labels else 0)

    unique_labels = set(labels)
    colors = [plt.cm.Spectral(each) for each in np.linspace(0, 1, len(unique_labels))]
    
    fig = plt.figure()
    ax = plt.axes()
    
    for k, col in zip(unique_labels, colors):
        if k == -1:
        # Black used for noise.
            col = [0, 0, 0, 1]

        class_member_mask = (labels == k)
        xy = X[class_member_mask & core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=10, picker = True)
        xy = X[class_member_mask & ~core_samples_mask]
        ax.plot(xy[:, 0], xy[:, 1], 'o', markerfacecolor=tuple(col), markeredgecolor='k', markersize=5, picker = True)
    
    fig.canvas.mpl_connect('pick_event', onPick)
    plt.title('Estimated number of clusters: %d' % n_clusters_)
    plt.xlabel('Principal Component 1')
    plt.ylabel('Principal Component 2')
    
    plt.show()
    return labels

def cluster_with_kmeans(X):
    kmeans = KMeans(n_clusters = 4)
    kmeans.fit(X)
    centroids = kmeans.cluster_centers_
    labels = kmeans.labels_
    
    colors = ["g.","k.","c.","b."]

    fig = plt.figure()
    ax = plt.axes()

    for i in range(len(X)):
        ax.plot(X[i][0], X[i][1], colors[labels[i]], markersize=10,alpha=0.5)
    ax.scatter(centroids[:,0], centroids[:, 1], marker='x', s=600, linewidths=20, c='red')
    plt.show()

def classify_with_knn(X,Y):
    #Cross Validation: apart our data into train and test data with %90 for train, %10 for test.
    x_train, x_test, y_train, y_test = train_test_split(X, Y, test_size=0.10)

    #train test data and learn the classes for features
    knn = neighbors.KNeighborsClassifier()
    knn.fit(x_train, y_train)

    #predict the classes regarding test data and compare it with real classes
    print("predict accuracy : ")
    accuracy = knn.score(x_test, y_test)
    print("accuracy : %" + str(accuracy * 100))

    #create two different input data and let
    example_data = np.array([[0.6, -0.2, -0.3, -0.4, 0.4, 0.1],[0.5, -0.1, -0.4, -0.3, 0.2, 0.0], [-0.1, -0.1, 0.2, 0.3, 0.1, -0.2]])
    prediction = knn.predict(example_data)
    print("predicted classes : ")
    print(prediction)

if __name__ == '__main__':
    # read data and transform it to numeric
    rawData, excelData = prepareData()
    
    #column "Zaman" possible does not have any effect on clustering or
    #classification.
    data = rawData.drop(columns=['Zaman'], axis=1)

    #normalize data in order to get rid of anomalies.  ( x1 - min(X) ) / #max(X) - min(X) )
    X = preprocessing.minmax_scale(data)

    #choose component count according to %90 variance ratio
    k = choose_k_dimension(X)

    #dimensionality reduction with k dimension
    X = doPCA(X, k)

    #draw the new data set on new components and see the clusters
    plot2dData(X)

    # plot 2d data with colors
    plot2dDataByColoredFeatures(X)

    #draw with the 3rd dimension
    plot3dData(X)

    #apply standart scale and plot data by colored points with Target data
    scaleAndTransformByTarget(X)

    #cluster data with DBSCAN (Density Based Spatial Clustering of Applications with Noise)
    Y = cluster_with_DBScan(X)

    #cluster data with KMeans
    cluster_with_kmeans(X)

    # classify data with KNN (k-nearest neighbors algorithm)
    classify_with_knn(X,Y)


