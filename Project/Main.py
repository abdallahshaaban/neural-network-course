from BackPropagation import TrainTheModel
from rbf import TrainTheModel_rbf
from PrepareDataset import GetDataset
import numpy as np
from MLP import MLP
import matplotlib.pyplot as plt
def Train():
    x_train,y_train,x_test,y_test = GetDataset("C:\\Users\\Lenovo-PC\\Desktop\\neural-network-course\\Project\\Data set\\Training","C:\\Users\\Lenovo-PC\\Desktop\\neural-network-course\\Project\\Data set\\Testing")
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
#    model = MLP()
    if AlgoVar.get():
        '''
        model.fit(x_train,y_train,epochs_Entry.get(),LearningRate_Entry.get(),Neurons_Entry.get(),Activation_Entry.get(),MSE_Entry.get(),var.get(),Hidden_Entry.get())
        predictions = model.predict(X=x_test)
        print(model.eval(predictions,y_test))
        print(predictions)
        
        plt.xlabel("Number of Epochs")
        plt.ylabel("Cost")
        plt.plot(model.epochs, model.cost)
        plt.show()
        '''
        TrainTheModel(Hidden_Entry.get(),epochs_Entry.get(),LearningRate_Entry.get(),Neurons_Entry.get(),Activation_Entry.get(),MSE_Entry.get(),var.get() , x_train,y_train,x_test,y_test)
    else:
        TrainTheModel_rbf(Neurons_Entry.get(),LearningRate_Entry.get(),MSE_Entry.get(),epochs_Entry.get(),5, x_train,y_train,x_test,y_test)
        



#x_train,y_train,x_test,y_test , data = GetDataset("C:\\Users\\Lenovo-PC\\Desktop\\neural-network-course\\Project\\Data set\\Training","C:\\Users\\Lenovo-PC\\Desktop\\neural-network-course\\Project\\Data set\\Testing")









































#------------------------------------ GUI ------------------------------------
from tkinter import *

#Creating the main window
root = Tk()
#Controls

Train_Button = Button(root , text = "Train The Model" , command = Train)
LearningRate_Label = Label(root , text = "learning rate")
LearningRate_Entry = Entry(root)
epochs_Label = Label(root , text = "number of epochs")
epochs_Entry = Entry(root)
Hidden_Label = Label(root , text = "number of hidden layers")
Hidden_Entry = Entry(root)
Neurons_Label = Label(root , text = "number of neurons per each layer")
Neurons_Entry = Entry(root)
Activation_Label = Label(root , text = "Activation function type")
Activation_Entry = Entry(root)
var = IntVar()
cBox = Checkbutton(root , text = "Use bias" , variable = var)
AlgoVar = IntVar()
AlgorithmCBox = Checkbutton(root , text = "Checked(MLP)/Not Checked(RBF)" , variable = AlgoVar)
MSE_Label = Label(root , text = "MSE threshold")
MSE_Entry = Entry(root)
#Controls' positions
LearningRate_Label.grid(row=2 , column=0 )
LearningRate_Entry.grid(row=2 , column=1 )
epochs_Label.grid(row=1 , column=0 )
epochs_Entry.grid(row=1 , column=1 )
Train_Button.grid(row=8, column=1)
MSE_Label.grid(row = 5 , column = 0)
MSE_Entry.grid(row = 5 , column = 1)
Hidden_Label.grid(row=0, column=0)
Hidden_Entry.grid(row=0, column=1)
Neurons_Label.grid(row=3, column=0)
Neurons_Entry.grid(row=3, column=1)
Activation_Label.grid(row=4, column=0)
Activation_Entry.grid(row=4, column=1)
cBox.grid(row=6, column=1)
AlgorithmCBox.grid(row=7,column=1)
#For Making the window still displayed
root.mainloop()
