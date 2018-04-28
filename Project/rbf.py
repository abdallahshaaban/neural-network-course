import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import math as m
 
def initialize_Centroid(k,x_data):
    centroids=[]
    for i in range(k):
        centroids.append(x_data[i])
    return centroids
def sigmoid(x):
        return 1 / (1 + np.exp(-x))
     
 
def Euclidean_distance(first, second):
    squDist = 0
    for i in range(len(first)):
            squDist += (first[i] - second[i])*(first[i] - second[i])
    euclDist = m.sqrt(squDist)
    return euclDist
 
 
KmeanThreshold=.0001
max_iterations=500
 
def kmean(k,x_data):
    centroids=initialize_Centroid(k,x_data)
    for i in range(max_iterations):
        clusters = {}
        for i in range(k):
            clusters[i] = []
        for features in x_data:
            distances = [Euclidean_distance(features ,centroids[centroid])  for centroid in range(len(centroids) ) ]
            classification = distances.index(min(distances))
            clusters[classification].append(features)
        previous = centroids
        for classification in clusters:
            centroids[classification] = np.average(clusters[classification], axis = 0)
        converge=True
        for centroid in range(len(centroids)):
            previous_centroid = previous[centroid]
            curr_centroid = centroids[centroid]
            if Euclidean_distance(curr_centroid , previous_centroid) < KmeanThreshold:
                converge = False
        if converge:        	
            break
    return centroids
 
 
 
def sigmaSpread(centroids,k):
    distance=[]
    for i in range(len(centroids)):
        for j in range(len(centroids)):
            if i == j:
                continue
            else: 
                distance.append(Euclidean_distance(centroids[i],centroids[j]))
 
    sigma=distance[distance.index(max(distance))]/m.sqrt(2*k)
    return sigma
 
 
def compute_Gaussian_fun(feature,centroids,sigma):
    gaussian=[]
    for i in  range(len(centroids)):
        r=Euclidean_distance(feature,centroids[i])
        gaussian.append(m.exp(-(r**2/(2*sigma**2))))
 
    return gaussian
 
def update_weights(weights,eta,error,gaussian,classes):
 
    for k in range(classes):
        weights[k]=weights[k]+(eta*error*gaussian)
    return weights
 
 
def MSE(y_act , y_pred,classes):
    error=0
    for j in range(len(y_act)):
        for i in range(classes):
            error+=(y_act[j,i] - y_pred[j,i])**2
    #print(error)
    return error/(4*len(y_act)) 
 
 
 
def GradientDescent(Y,Ypred,Alpha,Xs,Cs,classes):
    NewCs = np.full((len(Xs[0,:]),classes),0)
    Subt = Y - Ypred
    dcs = Xs.T.dot(Subt)
    NewCs = Cs - (Alpha* dcs)
    return NewCs
 
def TrainTheModel_rbf(Neurons_Entry,LearningRate_Entry,Mse_threshold,epochs_Entry,classes,x,y_train,x_test,y_test):
    global NumOfNeurons,eta,MseThreshold,Num_epochs,Num_of_classes,centroids,sigma,weights
    Num_of_classes=int(classes)
    Num_epochs=int(epochs_Entry)
 
    eta=float(LearningRate_Entry)
 
    NumOfNeurons=int(Neurons_Entry)
 
    MseThreshold=float(Mse_threshold)
    centroids=kmean(NumOfNeurons,x)
    sigma = sigmaSpread(centroids,NumOfNeurons)
    x_train= np.full((len(x) , NumOfNeurons) , 0.0)
    for i in range(len(x)):
        x_train[i] = compute_Gaussian_fun(x[i],centroids,sigma)
 
    weights = np.full((NumOfNeurons , Num_of_classes) , 0.0)
    import random
    for i in range(Num_of_classes):
        weights[:,i:i+1] = np.full((NumOfNeurons,1),random.uniform(0,1))
    prev=-1e10
    for i in range(Num_epochs):
        print(i)
        pred=x_train.dot(weights)
        pred=sigmoid(pred)
        #pprint.pprint(pred)
        #pprint.pprint(y_train)
        c=0
    
        for j in range(len(pred)):
            s=0
            m=-1e10
            for k in range(len(pred[j])):
                if pred[j,k:k+1]>m:
                    m=pred[j,k:k+1]
                    s=k

            #print(s)    
            if y_test[j,s]==1:
                c=c+1
        print("train : ",c/len(x_train))    
       
        mse=MSE(pred,y_train,Num_of_classes)
        weights=GradientDescent(y_train,pred,eta,x_train,weights,Num_of_classes)
        a=Test(x_test,y_test)
        if a>=prev:
            prev = a
        else:
            break
#        print(mse)
        if mse <= MseThreshold:
            break;
import pprint            
def Test (x_t,y_test):
    global NumOfNeurons,eta,MseThreshold,Num_epochs,Num_of_classes,centroids,sigma,weights
    x_test= np.full((len(x_t) , NumOfNeurons) , 0.0)
    for i in range(len(x_t)):
        x_test[i] = compute_Gaussian_fun(x_t[i],centroids,sigma)
    pred=x_test.dot(weights)
    pred=sigmoid(pred)

    c=0
    
    for j in range(len(pred)):
        s=0
        m=-1e10
        for k in range(len(pred[j])):
            if pred[j,k:k+1]>m:
                m=pred[j,k:k+1]
                s=k

            #print(s)    
        if y_test[j,s]==1:
                c=c+1
            
    print("test : ",c/len(x_test))    
  
    return c/len(x_test)
        
        
        