import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)
def sigmoid(x):
        return 1 / (1 + np.exp(-x))
    
def Dsigmoid(x):
        return x * (1 - x)
    
def DHyper_bolic_Tangent(x):
        return (1-(x**2))

def Hyper_bolic_Tangent(x):
    return (1 - np.exp(-x))/ (1 + np.exp(-x))


def SigmaError( y , y_hat ):
   global  Activations,Sigmas,Weights,Bias,Num_Hidden_Layer,Num_epochs,eta,Num_of_Neurons,Activation,Mse_threshold,Use_Bias
   Errors=[]
   sum=0
   for j in range(0,3):
        if(Activation=='sigmoid'):
           cost =  np.float((y[0][j] - y_hat[0][j]) * (y_hat[0][j]) * (1- y_hat[0][j]))
        else:
           cost =  np.float((y[0][j] - y_hat[0][j]) * (1-((y_hat[0][j])**2)))
        sum+=((y[0][j] - y_hat[0][j]))**2
        Errors.append(cost)
   return sum,Errors

                
def Feedforward(x,y): 
    
    for i in range(0,len(x)):
         a = x[i].reshape((1,4))
         Activations[0] = a
       
         """forward"""
         for l in range( 1 , Num_Hidden_Layer+2):
               if(Activation=='sigmoid'):
                 a = sigmoid(np.dot(a,Weights[l])+Bias[l] )
               else:
                 a = Hyper_bolic_Tangent(np.dot(a,Weights[l])+Bias[l]) 
               Activations[l] = a
               if(l==Num_Hidden_Layer+1):
                 Y = y[i].reshape((1,3))
                 output = Activations[l].reshape((1,3))
                 cost,sigma = SigmaError( Y, output )
                 sigma = np.array(sigma)
                 Sigmas[l] = sigma.reshape((1,3))   
         Backward(x,y)
         Update_Weight()
         
                 

def Backward(x,y):
    global  Activations,Sigmas,Weights,Bias,Num_Hidden_Layer,Num_epochs,eta,Num_of_Neurons,Activation,Mse_threshold,Use_Bias      
    for l in range( Num_Hidden_Layer , 0,-1):
                if(Activation=='sigmoid'):
                  da = Dsigmoid(Activations[l])
                else:
                  da = DHyper_bolic_Tangent(Activations[l]) 
                Sigmas[l] = np.dot(Sigmas[l+1],Weights[l+1].T) * da
                
                
def Update_Weight():
          global  Activations,Sigmas,Weights,Bias,Num_Hidden_Layer,Num_epochs,eta,Num_of_Neurons,Activation,Mse_threshold,Use_Bias
          for l in range( 1 ,Num_Hidden_Layer+2):
             Weights[l] = Weights[l] + (Sigmas[l] * eta *  Activations[l-1].T)
             if(Use_Bias==True):
               Bias[l] = Bias[l] + (Sigmas[l] * eta )


def Train(x,y):
           
           global  Activations,Sigmas,Weights,Bias,Num_Hidden_Layer,Num_epochs,eta,Num_of_Neurons,Activation,Mse_threshold,Use_Bias,x_train,y_train,x_test,y_test
           
           ys = []
           xs = []
           if(Mse_threshold ==0):
               for epoch in range(0,Num_epochs):
                  Feedforward(x,y)
                  confusion,cost= Mse(x_train,y_train )
                  ys.append(cost)
                  xs.append(epoch)
                  if(epoch%100==0):
                    Confusion,Mean_square_error = Mse(x_train,y_train)
                    #print(Confusion)
                    #print("Mse : " , Mean_square_error)
               Confusion,Mean_square_error = Mse(x_test,y_test)
               print(Confusion)
               print("Mse : " , Mean_square_error)
           else:
               j = 0
               Mean_square_error=1000
               while  Mean_square_error > Mse_threshold:
                    Feedforward(x,y)
                    Confusion,Mean_square_error = Mse(x_train,y_train)
                    ys.append(Mean_square_error)
                    xs.append(j)
                    j+=1
                    #print(Confusion)
                    #print("Mse : " , Mean_square_error)
                    
           ax1.clear()        
           ax1.plot(xs,ys)  
           plt.Show()      
           Confusion,Mean_square_error = Mse(x_test,y_test)
           print(Confusion)
           print("Accuracy : ", (Confusion[0][0]+Confusion[1][1]+Confusion[2][2])/60.0  )
                
                 

def Mse(x,y):
   global  Activations,Sigmas,Weights,Bias,Num_Hidden_Layer,Num_epochs,eta,Num_of_Neurons,Activation,Mse_threshold,Use_Bias
   Tot = 0
   m = len(x)
   Confusion = np.zeros((3,3))
   for i in range(0,len(x)):
      a = x[i].reshape((1,4))
      """forward"""
      for l in range( 1 , Num_Hidden_Layer+2):
          if(Activation=='sigmoid'):
             a = sigmoid(np.dot(a,Weights[l])+ Bias[l])
          else:
             a = Hyper_bolic_Tangent(np.dot(a,Weights[l])+Bias[l])  
          if(l==Num_Hidden_Layer+1):
              Y = y_train[i].reshape((1,3))
              output = a.reshape((1,3))
              idx = [ j for j in range(0,3) if y[i][j]==1]
              ind = np.argmax(a[0], axis=0)
              Confusion[idx[0]][ind]+=1
              #print(output)
              cost,sigma = SigmaError( Y, output )
              Tot+=cost
  
   return Confusion , (1/(2*m))*Tot
            
def LoadData():
    
    dataset = pd.read_csv('Iris.txt' , sep=",")
    x_data = np.array(dataset.iloc[:, 0:4].values)
    y_val = dataset.iloc[:, 4:5].values 
    
    y_true = np.zeros((150,3))
    
    for i in range(0,150):
        if(y_val[i]=='Iris-setosa'):
           y_true[i][0]=1
        elif(y_val[i]=='Iris-versicolor'):
           y_true[i][1]=1
        elif(y_val[i]=='Iris-virginica'):
           y_true[i][2]=1
         
    x_train = np.concatenate( (x_data[0:30 , :] , x_data[50:80 , :] ,x_data[100:130 , :]) ,axis=0)
    y_train = np.concatenate( (y_true[0:30 , :] , y_true[50:80 , :] ,y_true[100:130 , :]) ,axis=0)
    
    x_test =  np.concatenate( (x_data[30:50 , :] , x_data[80:100 , :] ,x_data[130:150 , :]) , axis=0 )
    y_test =  np.concatenate( (y_true[30:50 , :] , y_true[80:100 , :] ,y_true[130:150 , :]) , axis=0 )
    return x_train,y_train,x_test,y_test



def TrainTheModel():
    '''
    put the whole code here (training ---> Testing ---> Show the graph)
    this function will be called using "Train The Model" button
    '''
    global  Activations,Sigmas,Weights,Bias,Num_Hidden_Layer,Num_epochs,eta,Num_of_Neurons,Activation,Mse_threshold,Use_Bias,x_train,y_train,x_test,y_test
    Num_Hidden_Layer = int(Hidden_Entry.get()) 
    Num_epochs = int(epochs_Entry.get()) 
    eta = float(LearningRate_Entry.get()) 
    Num_of_Neuron=str(Neurons_Entry.get())  #you must split to get the actual values
    Activation = str(Activation_Entry.get()) 
    Mse_threshold = float(MSE_Entry.get()) 
    
    if var.get():
        Use_Bias = True
    else:
        Use_Bias = False
    
    Num_of_Neurons = []
    Num_of_Neurons.append(4)
    for neuron in Num_of_Neuron.split():
      Num_of_Neurons.append(int(neuron))   
    Num_of_Neurons.append(3)
    
    print(type(Num_of_Neurons))
    
    x_train,y_train,x_test,y_test = LoadData()

    Weights = {}
    Bias = {}
    for l in range( 1 , Num_Hidden_Layer+2):
        Weights[l] = np.random.randn(Num_of_Neurons[l-1], Num_of_Neurons[l])*0.01
        Bias[l] = np.zeros(( 1 ,Num_of_Neurons[l]))
        
    Activations = {}
    Sigmas = {}
    
    Train(x_train,y_train)
    
    
#------------------------------------ GUI ------------------------------------
from tkinter import *

#Creating the main window
root = Tk()
#Controls

Train_Button = Button(root , text = "Train The Model" , command = TrainTheModel)
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
MSE_Label = Label(root , text = "MSE threshold")
MSE_Entry = Entry(root)
#Controls' positions
LearningRate_Label.grid(row=2 , column=0 )
LearningRate_Entry.grid(row=2 , column=1 )
epochs_Label.grid(row=1 , column=0 )
epochs_Entry.grid(row=1 , column=1 )
Train_Button.grid(row=7, column=1)
MSE_Label.grid(row = 5 , column = 0)
MSE_Entry.grid(row = 5 , column = 1)
Hidden_Label.grid(row=0, column=0)
Hidden_Entry.grid(row=0, column=1)
Neurons_Label.grid(row=3, column=0)
Neurons_Entry.grid(row=3, column=1)
Activation_Label.grid(row=4, column=0)
Activation_Entry.grid(row=4, column=1)
cBox.grid(row=6, column=1)
#For Making the window still displayed
root.mainloop()


