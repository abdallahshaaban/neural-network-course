from BackPropagation import MLP
from rbf import TrainTheModel_rbf
from PrepareDataset import Preparation
import numpy as np
import matplotlib.pyplot as plt
def Train():
<<<<<<< HEAD
    global MLPObj , PrepareObj
    MLPObj = MLP()
    PrepareObj = Preparation()
    x_train,y_train,x_test,y_test = PrepareObj.GetDataset("C:\\Users\\Lenovo-PC\\Desktop\\neural-network-course\\Project\\Data set\\Training","C:\\Users\\Lenovo-PC\\Desktop\\neural-network-course\\Project\\Data set\\Testing")
=======
    x_train,y_train,x_test,y_test = GetDataset("C:\\Users\\abdal_000\\Documents\\GitHub\\neural-network-course\\Project\\Data set\\Training","C:\\Users\\abdal_000\\Documents\\GitHub\\neural-network-course\\Project\\Data set\\Testing")
>>>>>>> 5ac0b7479892a188a96624517d723c946bd79a80
    from sklearn.preprocessing import StandardScaler
    sc_x = StandardScaler()
    x_train = sc_x.fit_transform(x_train)
    x_test = sc_x.transform(x_test)
    if AlgoVar.get():
        MLPObj.TrainTheModel(Hidden_Entry.get(),epochs_Entry.get(),LearningRate_Entry.get(),Neurons_Entry.get(),Activation_Entry.get(),MSE_Entry.get(),var.get() , x_train,y_train,x_test,y_test)
    else:
        TrainTheModel_rbf(Neurons_Entry.get(),LearningRate_Entry.get(),MSE_Entry.get(),epochs_Entry.get(),5, x_train,y_train,x_test,y_test)
        

def OpenImage():
    from PIL import ImageTk, Image
    canvas.image = ImageTk.PhotoImage(Image.open(str(ImageName_Entry.get()) + ".jpg"))
    canvas.create_image(0,0,anchor = 'nw' , image = canvas.image)
def Classify():
    global MLPObj , PrepareObj
    Features = PrepareObj.PrepareSample(str(ImageName_Entry.get()))
    if AlgoVar.get():
        Preds = MLPObj.Classify(Features , PrepareObj.classes)
    PrepareObj.Display(Preds , str(ImageName_Entry.get()))

#x_train,y_train,x_test,y_test , data = GetDataset("C:\\Users\\Lenovo-PC\\Desktop\\neural-network-course\\Project\\Data set\\Training","C:\\Users\\Lenovo-PC\\Desktop\\neural-network-course\\Project\\Data set\\Testing")









































#------------------------------------ GUI ------------------------------------
from tkinter import *

#Creating the main window
root = Tk()
#Controls
Hidden_Label = Label(root , text = "number of hidden layers")
Hidden_Entry = Entry(root)
epochs_Label = Label(root , text = "number of epochs")
epochs_Entry = Entry(root)
LearningRate_Label = Label(root , text = "learning rate")
LearningRate_Entry = Entry(root)
Neurons_Label = Label(root , text = "number of neurons per each layer")
Neurons_Entry = Entry(root)
Activation_Label = Label(root , text = "Activation function type")
Activation_Entry = Entry(root)
MSE_Label = Label(root , text = "MSE threshold")
MSE_Entry = Entry(root)
var = IntVar()
cBox = Checkbutton(root , text = "Use bias" , variable = var)
AlgoVar = IntVar()
AlgorithmCBox = Checkbutton(root , text = "Checked(MLP)/Not Checked(RBF)" , variable = AlgoVar)
Train_Button = Button(root , text = "Train The Model" , command = Train)
ImageName_Label = Label(root , text = "Image Name")
ImageName_Entry = Entry(root)
OpenImage_Button = Button(root , text = "Open The Image" , command = OpenImage)
Classify_Button = Button(root , text = "Classify" , command = Classify)
canvas = Canvas(root,width=500,height=500)

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
canvas.grid(row = 10 , column = 1)
ImageName_Label.grid(row=9,column=0)
ImageName_Entry.grid(row=9,column=1)
OpenImage_Button.grid(row=9 , column=2)
Classify_Button.grid(row = 11 , column = 1)
#For Making the window still displayed
root.mainloop()
