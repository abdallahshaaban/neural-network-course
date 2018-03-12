# -*- coding: utf-8 -*-
"""
Created on Sat Feb 24 12:01:05 2018

@author: Lenovo-PC
"""
from tkinter import *
from tkinter import messagebox
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd



# Import The Dataset
dataset = pd.read_csv('Iris Data.txt' , sep=",")
x = dataset.iloc[:, 0:4].values
y = dataset.iloc[:, 4:5].values 
# 1. Draw Iris dataset
def DrawTheData():
    L = np.matrix([[0,1],[0,2],[0,3],[1,2],[1,3],[2,3]])
    for i in range(len(L[:,0])):
        X1 = L[i,0]
        X2 = L[i,1]
        plt.scatter(x[0:51,X1],x[0:51,X2],color='red')
        plt.scatter(x[51:101,X1],x[51:101,X2],color='green')
        plt.scatter(x[101:,X1],x[101:,X2],color='blue')
        plt.title('Figure Of Data')
        plt.xlabel('X' + str(X1 + 1))
        plt.ylabel('X' + str(X2 + 1))
        plt.show()
        
        
# 2. Train The Model
x_train,x_test,y_train,y_test,cs = [[],[],[],[],[]]
def TrainTheModel():
    global x_train,x_test,y_train,y_test,cs
    X1 = int(Features_Entry1.get()) - 1
    X2 = int(Features_Entry2.get()) - 1
    C1 = int(Classes_Entry1.get()) - 1
    C2 = int(Classes_Entry2.get()) - 1
    epochs = int(epochs_Entry.get())
    alpha = float(LearningRate_Entry.get())
    MSEVal = float(MSE_Entry.get())
    X = np.concatenate( ( np.concatenate( (x[C1*50 : (C1+1)*50 , X1:X1+1] , x[C1*50 : (C1+1)*50 , X2:X2+1] ) , axis = 1) , np.concatenate( (x[C2*50 : (C2+1)*50 , X1:X1+1] , x[C2*50 : (C2+1)*50 , X2:X2+1] ) , axis = 1) ) , axis=0 )
    Y = np.concatenate( ( y[C1*50 : (C1+1)*50 , 0:1] , y[C2*50 : (C2+1)*50 , 0:1]  ) , axis=0 )
    #Encoding Categorical Data
    from sklearn.preprocessing import LabelEncoder 
    labelencoder_y = LabelEncoder()
    Y[:,0] = labelencoder_y.fit_transform(Y[:,0])
    x_train = np.concatenate( (X[0:30 , :] , X[50:80 , :]) , axis=0 )
    x_test = np.concatenate( (X[30:50 , :] , X[80: , :]) , axis=0 )
    y_train = np.concatenate( (Y[0:30 , :] , Y[50:80 , :]) , axis=0 )
    y_train[0:30 , 0] = -1
    y_test = np.concatenate( (Y[30:50 , :] , Y[80: , :]) , axis=0 )
    y_test[0:20 , 0] = -1
    if var.get():
        x_train = np.append(x_train , np.full((len(x_train[:,0]),1),1) ,1)
        x_test = np.append(x_test , np.full((len(x_test[:,0]),1),1) ,1)
    import random
    cs=np.full((len(x_train[0,:]),1),random.uniform(0,1))
    for i in range(epochs):
        YPred = x_train.dot(cs)
        cs = GradientDescent(y_train,YPred,alpha,x_train,cs)
        CurrMSE = MSE(y_train , YPred)
        if CurrMSE <= MSEVal:
            break;
    messagebox.showinfo("Info", "The Training Done Correctly (#Epochs: " + str(i+1) + ", MSE: " + str(CurrMSE))
    
# 3.Draw a line that can discriminate between the two learned classes
def DrawLine():
    a = Features_Entry1.get()
    b = Features_Entry2.get()
    C1 = int(Classes_Entry1.get()) - 1
    C2 = int(Classes_Entry2.get()) - 1
    plt.scatter(x_train[0:30,0],x_train[0:30,1],color='red')
    plt.scatter(x_train[30:60,0],x_train[30:60,1],color='green')
    if var.get():
        plt.plot(x_train[:,0] , -1 * ((x_train[:,0].dot(cs[0,0]) + x_train[:,2].dot(cs[2,0]))/cs[1,0]))
    else:
        plt.plot(x_train[:,0] , -1 * ((x_train[:,0].dot(cs[0,0]))/cs[1,0]))
    plt.title('Figure Of Data')
    plt.xlabel('X' + a)
    plt.ylabel('X' + b)
    plt.show()
    
# 4.Test the classifier
ConfMatrix = []
Ypred=[]
def TestTheClassifier():
    global Ypred , ConfMatrix
    ConfMatrix = np.full((2,2),0)
    Ypred = signum(x_test,cs)
    for i in range(len(Ypred[:,0])):
        if y_test[i,0] == -1 and Ypred[i,0] == -1:
            ConfMatrix[0,0] +=1
        elif y_test[i,0] == -1 :
            ConfMatrix[0,1] +=1
        elif Ypred[i,0] == -1:
            ConfMatrix[1,0] +=1
        else:
            ConfMatrix[1,1] +=1
    Accuracy = ((ConfMatrix[0,0]+ConfMatrix[1,1]) / len(Ypred[:,0])) * 100
    messagebox.showinfo("Info", "The accuracy is " + str(Accuracy) + "%")
    
#Perceptron Model

def signum(Xs,Cs):
    XC = Xs.dot(Cs)
    Results = np.zeros((len(XC[:,0]),1))
    for i in range(len(XC[:,0])):
        if XC[i,0] > 0 :
            Results[i,0] = 1
        elif XC[i,0] < 0 :
            Results[i,0] = -1
        else:
            Results[i,0] = 0
    return Results

def GradientDescent(Y,Ypred,Alpha,Xs,Cs):
    NewCs = np.full((len(Xs[0,:]),1),0)
    Subt = Y - Ypred
    dcs = Xs.T.dot(Subt)
    NewCs = Cs + (Alpha* dcs)
    return NewCs

def MSE(ActRes , NewRes):
    error = (ActRes - NewRes)**2
    return (np.dot(np.ones((1,len(ActRes[:,0]))),error)/(2*float(len(ActRes[:,0]))))
#------------------------------------ GUI ------------------------------------

#Creating the main window
root = Tk()
#Controls
Features_Label = Label(root , text = "Features")
Features_Entry1 = Entry(root)
Features_Entry2 = Entry(root)
Features_AndLabel = Label(root , text = "And")
Draw_Button = Button(root , text = "Draw Data" , command = DrawTheData)
Classes_Label = Label(root , text = "Classes")
Classes_AndLabel = Label(root , text = "And")
Classes_Entry1 = Entry(root)
Classes_Entry2 = Entry(root)
Train_Button = Button(root , text = "Train The Model" , command = TrainTheModel)
LearningRate_Label = Label(root , text = "learning rate")
LearningRate_Entry = Entry(root)
epochs_Label = Label(root , text = "number of epochs")
epochs_Entry = Entry(root)
var = IntVar()
cBox = Checkbutton(root , text = "Use bias" , variable = var)
DrawLine_Button = Button(root , text = "Draw Line" , command = DrawLine)
Test_Button = Button(root , text = "Test The Model" , command = TestTheClassifier)
MSE_Label = Label(root , text = "MSE")
MSE_Entry = Entry(root)
#Controls' positions
Features_Label.grid(row=0 , column=0)
Features_Entry1.grid(row=0 , column=1)
Features_AndLabel.grid(row=0 , column=2)
Features_Entry2.grid(row=0 , column=3)
Draw_Button.grid(row = 0 , column = 4)
Classes_Label.grid(row=1 , column=0)
Classes_Entry1.grid(row=1 , column=1 )
Classes_AndLabel.grid(row=1 , column=2 )
Classes_Entry2.grid(row=1 , column=3 )
LearningRate_Label.grid(row=2 , column=0 )
LearningRate_Entry.grid(row=2 , column=1 )
epochs_Label.grid(row=3 , column=0 )
epochs_Entry.grid(row=3 , column=1 )
Train_Button.grid(row=1, column=4)
cBox.grid(row=5, column=1)
DrawLine_Button.grid(row=2 , column=4 )
Test_Button.grid(row=3 , column=4 )
MSE_Label.grid(row = 4 , column = 0)
MSE_Entry.grid(row = 4 , column = 1)
#For Making the window still displayed
root.mainloop()


