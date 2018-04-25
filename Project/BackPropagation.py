import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import copy
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
   for j in range(0,5):
        if(Activation=='sigmoid'):
           cost =  np.float((y[0][j] - y_hat[0][j]) * (y_hat[0][j]) * (1- y_hat[0][j]))
        else:
           cost =  np.float((y[0][j] - y_hat[0][j]) * (1-((y_hat[0][j])**2)))
        sum+=((y[0][j] - y_hat[0][j]))**2
        Errors.append(cost)
   return sum,Errors

                
def Feedforward(x,y): 
    
    for i in range(0,len(x)):
         a = x[i].reshape((1,23))
         Activations[0] = a
       
         """forward"""
         for l in range( 1 , Num_Hidden_Layer+2):
               if(Activation=='sigmoid'):
                 a = sigmoid(np.dot(a,Weights[l])+Bias[l] )
               else:
                 a = Hyper_bolic_Tangent(np.dot(a,Weights[l])+Bias[l]) 
               Activations[l] = a
               if(l==Num_Hidden_Layer+1):
                 Y = y[i].reshape((1,5))
                 output = Activations[l].reshape((1,5))
                 cost,sigma = SigmaError( Y, output )
                 sigma = np.array(sigma)
                 Sigmas[l] = sigma.reshape((1,5))   
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
           PrevACC = -1
           PrevWeights = copy.deepcopy(Weights)
           BestAcc = []
           ys = []
           xs = []
           if(Mse_threshold == 0):
               for epoch in range(0,Num_epochs):
                  Feedforward(x,y)
                  confusion,cost= Mse(x_train,y_train )
                  ys.append(cost)
                  xs.append(epoch)
                  if(epoch%10==0):
                    Confusion,Mean_square_error = Mse(x_train,y_train)
                    print(Confusion)
                    print("Mse : " , Mean_square_error)
                    TestConf,TestMSE = Mse(x_test,y_test)
                    Acc = (TestConf[0][0]+TestConf[1][1]+TestConf[2][2] + TestConf[3][3] + TestConf[4][4])/26.0
                    print(TestConf)
                    print("\n",Acc,"\n---------------------------------------------------\n")
                    BestAcc.append(Acc)
                    '''
                    if Acc < PrevACC:
                        Weights = copy.deepcopy(PrevWeights)
                        break;
                    PrevACC = Acc
                    PrevWeights = copy.deepcopy(Weights)
                    '''
           else:
               j = 0
               Mean_square_error=1000
               while  Mean_square_error > Mse_threshold:
                    Feedforward(x,y)
                    Confusion,Mean_square_error = Mse(x_train,y_train)
                    ys.append(Mean_square_error)
                    xs.append(j)
                    j+=1
                    if(j%10==0):
                        Confusion,Mean_square_error = Mse(x_train,y_train)
                        print(Confusion)
                        print("Mse : " , Mean_square_error)
                        TestConf,TestMSE = Mse(x_test,y_test)
                        Acc = (TestConf[0][0]+TestConf[1][1]+TestConf[2][2] + TestConf[3][3] + TestConf[4][4])/26.0
                        print(TestConf)
                        print("\n",Acc,"\n---------------------------------------------------\n")
                        BestAcc.append(Acc)
                        '''
                        if Acc < PrevACC:
                            Weights = copy.deepcopy(PrevWeights)
                            break;
                        PrevACC = Acc
                        PrevWeights = copy.deepcopy(Weights)
                        '''
                        
           fig = plt.figure()
           ax1 = fig.add_subplot(1,1,1)         
           ax1.clear()        
           ax1.plot(xs,ys)  
           plt.show()     
           Confusion,Mean_square_error = Mse(x_test,y_test)
           print(Confusion)
           print("Mse : " , Mean_square_error)
           print("Accuracy : ", (Confusion[0][0]+Confusion[1][1]+Confusion[2][2] + Confusion[3][3] + Confusion[4][4])/26.0  )
           print("The best accuracy is: " ,max(BestAcc))     
                 

def Mse(x,y):
   global  Activations,Sigmas,Weights,Bias,Num_Hidden_Layer,Num_epochs,eta,Num_of_Neurons,Activation,Mse_threshold,Use_Bias
   Tot = 0
   m = len(x)
   Confusion = np.zeros((5,5))
   for i in range(len(x)):
      a = x[i].reshape((1,23))
      """forward"""
      for l in range( 1 , Num_Hidden_Layer+2):
          if(Activation=='sigmoid'):
             a = sigmoid(np.dot(a,Weights[l])+ Bias[l])
          else:
             a = Hyper_bolic_Tangent(np.dot(a,Weights[l])+Bias[l])  
          if(l==Num_Hidden_Layer+1):
              Y = y[i].reshape((1,5))
              output = a.reshape((1,5))
              idx = [ j for j in range(0,5) if y[i][j]==1]
              ind = np.argmax(a[0], axis=0)
              Confusion[idx[0]][ind]+=1
              #print(output)
              cost,sigma = SigmaError( Y, output )
              Tot+=cost
  
   return Confusion , (1/(2*m))*Tot
            

def TrainTheModel(Hidden_Entry,epochs_Entry,LearningRate_Entry,Neurons_Entry,Activation_Entry,MSE_Entry,var,x_tr,y_tr,x_ts,y_ts):
    '''
    put the whole code here (training ---> Testing ---> Show the graph)
    this function will be called using "Train The Model" button
    '''
    global  Activations,Sigmas,Weights,Bias,Num_Hidden_Layer,Num_epochs,eta,Num_of_Neurons,Activation,Mse_threshold,Use_Bias,x_train,y_train,x_test,y_test
    Num_Hidden_Layer = int(Hidden_Entry) 
    Num_epochs = int(epochs_Entry) 
    eta = float(LearningRate_Entry) 
    Num_of_Neuron=str(Neurons_Entry)  #you must split to get the actual values
    Activation = str(Activation_Entry) 
    Mse_threshold = float(MSE_Entry) 
    x_train = x_tr
    y_train = y_tr
    x_test = x_ts
    y_test = y_ts
    if var:
        Use_Bias = True
    else:
        Use_Bias = False
    
    Num_of_Neurons = []
    Num_of_Neurons.append(len(x_train[0,:]))
    for neuron in Num_of_Neuron.split(","):
      Num_of_Neurons.append(int(neuron))   
    Num_of_Neurons.append(len(y_train[0,:]))
    
    print(type(Num_of_Neurons))
    
    Weights = {}
    Bias = {}
    for l in range( 1 , Num_Hidden_Layer+2):
        Weights[l] = np.random.randn(Num_of_Neurons[l-1], Num_of_Neurons[l])*0.01
        Bias[l] = np.zeros(( 1 ,Num_of_Neurons[l]))
        
    Activations = {}
    Sigmas = {}
    
    Train(x_train,y_train)
    
    
