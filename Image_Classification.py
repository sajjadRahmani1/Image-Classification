# -*- coding: utf-8 -*-
import tensorflow as tf
import numpy as np
#import torch
import seaborn as sns
#%matplotlib inline
import matplotlib.pyplot as plt
import time
np.random.seed(1234)

(x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()

# Rescale the images from [0,255] to the [0.0,1.0] range. (Normalize)
x_train, x_test = x_train[..., np.newaxis]/255.0, x_test[..., np.newaxis]/255.0
# non normalizing will reduce accuracy, changing speed won't help it.


print("Number of original training examples:", len(x_train))
print("Number of original test examples:", len(x_test))

# Vectorization
x_train = x_train.reshape(x_train.shape[0],x_train.shape[1]*x_train.shape[2])
x_test = x_test.reshape(x_test.shape[0],x_test.shape[1]*x_test.shape[2])


# Change y_train and y_test
ytemp = np.zeros((y_train.shape[0], 10))
for i in range(y_train.shape[0]):
    ytemp[i, y_train[i]] = 1
y_train = ytemp

ytemp = np.zeros((y_test.shape[0], 10))
for i in range(y_test.shape[0]):
    ytemp[i, y_test[i]] = 1
y_test = ytemp
    

# Recource: https://mlfromscratch.com/neural-network-tutorial/#/
# The initial algorithm was derived from here and manipulated according to our needs
class MLP():
    def __init__(self, sizes=[784, 128, 128, 10], epochs=10, learningRate=0.001, numHidLayer=2, actFunc="ReLu", outActFunc="softmax", L2Reg = False, lamda=0.0001 , bias = False): #sajjad
        self.sizes = sizes
        self.epochs = epochs
        self.learningRate = learningRate
        self.numHidLayer = numHidLayer
        self.actFunc = actFunc
        self.outActFunc = outActFunc
        self.L2Reg = L2Reg
        self.lamda = lamda
        self.bias = bias #sajjad

        # we save all parameters in the neural network in this dictionary
        self.p = self.initialization(numHidLayer = self.numHidLayer)

    def sigmoid(self, x, derivative=False):
        if derivative:
            return (np.exp(-x))/((np.exp(-x)+1)**2)
        return 1/(1 + np.exp(-x))

    def softmax(self, x, derivative=False):
        # Numerically stable with large exponentials
        exps = np.exp(x - x.max())
        if derivative:
            return exps / np.sum(exps, axis=0) * (1 - exps / np.sum(exps, axis=0))
        return exps / np.sum(exps, axis=0)
    
    def relu(self, x, derivative=False):
        #print("=>", type(x))
        if derivative:
            x[x<=0] = 0
            x[x>0] = 1
            return x
        return np.maximum(x, 0)
    
    def tanh(self, x, derivative=False): 
        t = (np.exp(x)-np.exp(-x))/(np.exp(x)+np.exp(-x))
        if derivative:
            return 1-t**2
        return t

    def initialization(self, numHidLayer):
        '''# number of nodes in each layer
        input_layer=self.sizes[0]
        hidden_1=self.sizes[1]
        hidden_2=self.sizes[2]
        output_layer=self.sizes[3]'''
        if (self.bias == True):   
           #sajjad
           if (numHidLayer == 0):
            self.sizes[0] += 1
           if (numHidLayer == 1):
            self.sizes[0] += 1 ; self.sizes[1] += 1 ; 
           if (numHidLayer == 2):
            self.sizes[0] += 1 ; self.sizes[1] += 1 ; self.sizes[2] += 1 
           if (numHidLayer == 3):     
            self.sizes[0] += 1 ; self.sizes[1] += 1 ; self.sizes[2] += 1 ; self.sizes[3] += 1 
                  
        if (numHidLayer == 0):
            # number of nodes in each layer
            input_layer=self.sizes[0]
            output_layer=self.sizes[1]
            p = {
                'W1':np.random.randn(output_layer, input_layer) * np.sqrt(1. / output_layer)
                }
        elif (numHidLayer == 1):
            input_layer=self.sizes[0]
            hidden_1=self.sizes[1]
            output_layer=self.sizes[2]
            p = {
                'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
                'W2':np.random.randn(output_layer, hidden_1) * np.sqrt(1. / output_layer),
            }
        elif (numHidLayer == 2):
            input_layer=self.sizes[0]
            hidden_1=self.sizes[1]
            hidden_2=self.sizes[2]
            output_layer=self.sizes[3]
            #print("11::",input_layer,  hidden_1,  hidden_2, output_layer)
            
            p = {
                'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
                'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
                'W3':np.random.randn(output_layer, hidden_2) * np.sqrt(1. / output_layer)
            }
        elif (numHidLayer == 3):
            input_layer=self.sizes[0]
            hidden_1=self.sizes[1]
            hidden_2=self.sizes[2]
            hidden_3=self.sizes[3]
            output_layer=self.sizes[4]
            p = {
                'W1':np.random.randn(hidden_1, input_layer) * np.sqrt(1. / hidden_1),
                'W2':np.random.randn(hidden_2, hidden_1) * np.sqrt(1. / hidden_2),
                'W3':np.random.randn(hidden_3, hidden_2) * np.sqrt(1. / hidden_3),
                'W4':np.random.randn(output_layer, hidden_3) * np.sqrt(1. / output_layer)
            }
        
        return p

    def forward_propagation(self, x_train, numHidLayer, actFunc, outActFunc):
        p = self.p
        
        if (numHidLayer==0):
            # input layer activations becomes sample
            p['A0'] = x_train
            # input layer to outputLayer
            p['Z1'] = np.dot(p["W1"], p['A0'])
            p['A1'] = outActFunc(p['Z1'])
            return p['A1']
        
        elif (numHidLayer==1):
            # input layer activations becomes sample
            p['A0'] = x_train
            # input layer to hidden layer 1
            p['Z1'] = np.dot(p["W1"], p['A0'])
            p['A1'] = actFunc(p['Z1'])
            # hidden layer 1 to outputlayer
            p['Z2'] = np.dot(p["W2"], p['A1'])
            p['A2'] = outActFunc(p['Z2'])
            return p['A2']
            
        elif (numHidLayer==2):
            # input layer activations becomes sample

            p['A0'] = x_train
            # input layer to hidden layer 1
            p['Z1'] = np.dot(p["W1"], p['A0'])
            p['A1'] = actFunc(p['Z1'])
            # hidden layer 1 to hidden layer 2
            p['Z2'] = np.dot(p["W2"], p['A1'])
            p['A2'] = actFunc(p['Z2'])
            # hidden layer 2 to output layer
            p['Z3'] = np.dot(p["W3"], p['A2'])
            p['A3'] = outActFunc(p['Z3'])
            return p['A3']
            
        elif(numHidLayer==3):
            # input layer activations becomes sample
            p['A0'] = x_train
            # input layer to hidden layer 1
            p['Z1'] = np.dot(p["W1"], p['A0'])
            p['A1'] = actFunc(p['Z1'])
            # hidden layer 1 to hidden layer 2
            p['Z2'] = np.dot(p["W2"], p['A1'])
            p['A2'] = actFunc(p['Z2'])
            # hidden layer 2 to hidden layer 3
            p['Z3'] = np.dot(p["W3"], p['A2'])
            p['A3'] = actFunc(p['Z3'])
            # hidden layer 3 to outputlayer
            p['Z4'] = np.dot(p["W4"], p['A3'])
            p['A4'] = actFunc(p['Z4'])
            return p['A4']
            
        
        '''# input layer activations becomes sample
        p['A0'] = x_train
        # input layer to hidden layer 1
        p['Z1'] = np.dot(p["W1"], p['A0'])
        p['A1'] = self.sigmoid(p['Z1'])
        # hidden layer 1 to hidden layer 2
        p['Z2'] = np.dot(p["W2"], p['A1'])
        p['A2'] = self.sigmoid(p['Z2'])
        # hidden layer 2 to output layer
        p['Z3'] = np.dot(p["W3"], p['A2'])
        p['A3'] = self.softmax(p['Z3'])
        return p['A3']'''

    def backward_propagation(self, y_train, output, numHidLayer, actFunc, outActFunc):
        '''
            This is the backpropagation algorithm, for calculating the updates
            of the neural network's parameters.

            Note: There is a stability issue that causes warnings. This is 
                  caused  by the dot and multiply operations on the huge arrays.
                  
                  RuntimeWarning: invalid value encountered in true_divide
                  RuntimeWarning: overflow encountered in exp
                  RuntimeWarning: overflow encountered in square
        '''
        p = self.p
        change_weights = {}
        
        if (numHidLayer==0):
            # Calculate W1 update
            error = 2 * (output - y_train) / output.shape[0] * outActFunc(p['Z1'], derivative=True)
            change_weights['W1'] = np.outer(error, p['A0'])
            
        elif (numHidLayer==1):
             # Calculate W2 update
             error = 2 * (output - y_train) / output.shape[0] * outActFunc(p['Z2'], derivative=True)
             change_weights['W2'] = np.outer(error, p['A1'])
             # Calculate W1 update
             error = np.dot(p['W2'].T, error) * actFunc(p['Z1'], derivative=True)
             change_weights['W1'] = np.outer(error, p['A0'])
            
        
        elif (numHidLayer==2):
             # Calculate W3 update
             error = 2 * (output - y_train) / output.shape[0] * outActFunc(p['Z3'], derivative=True)
             change_weights['W3'] = np.outer(error, p['A2'])
             # Calculate W2 update
             error = np.dot(p['W3'].T, error) * actFunc(p['Z2'], derivative=True)
             change_weights['W2'] = np.outer(error, p['A1'])
             # Calculate W1 update 
             error = np.dot(p['W2'].T, error) * actFunc(p['Z1'], derivative=True)
             change_weights['W1'] = np.outer(error, p['A0'])
            
        elif (numHidLayer==3):
            # Calculate W4 update
            error = 2 * (output - y_train) / output.shape[0] * outActFunc(p['Z4'], derivative=True)
            change_weights['W4'] = np.outer(error, p['A3'])
            # Calculate W3 update
            error = np.dot(p['W4'].T, error) * actFunc(p['Z3'], derivative=True)
            change_weights['W3'] = np.outer(error, p['A2'])
            # Calculate W2 update
            error = np.dot(p['W3'].T, error) * actFunc(p['Z2'], derivative=True)
            change_weights['W2'] = np.outer(error, p['A1'])
            # Calculate W1 update
            error = np.dot(p['W2'].T, error) * actFunc(p['Z1'], derivative=True)
            change_weights['W1'] = np.outer(error, p['A0'])


        '''# Calculate W3 update
        error = 2 * (output - y_train) / output.shape[0] * self.softmax(p['Z3'], derivative=True)
        change_weights['W3'] = np.outer(error, p['A2'])

        # Calculate W2 update
        error = np.dot(p['W3'].T, error) * self.sigmoid(p['Z2'], derivative=True)
        change_weights['W2'] = np.outer(error, p['A1'])

        # Calculate W1 update
        error = np.dot(p['W2'].T, error) * self.sigmoid(p['Z1'], derivative=True)
        change_weights['W1'] = np.outer(error, p['A0'])'''
        
    
        # return ∇J(x, y)
        return change_weights

    def SGD(self, changesOfWeights):
        '''
            Update network parameters according to update rule from
            Stochastic Gradient Descent.

            θ = θ - η * ∇J(x, y), 
                theta θ:            a network parameter (e.g. a weight w)
                eta η:              the learning rate
                gradient ∇J(x, y):  the gradient of the objective function,
                                    i.e. the change for a specific theta θ
        '''
        
        for layer, changeWValue in changesOfWeights.items():
            if (self.L2Reg):
                L2RegValue = (self.lamda/2)*sum(self.p[layer]**2)
                self.p[layer] -= self.learningRate * (changeWValue+L2RegValue)
            else:
                self.p[layer] -= self.learningRate * changeWValue

    def evaluate_acc(self, x_test, y_test, numHidLayer, actFunc, outActFunc):
        '''
            This function does a forward pass of x, then checks if the indices
            of the maximum value in the output equals the indices in the label
            y. Then it sums over each prediction and calculates the accuracy.
        '''
        predictions = []

        for x, y in zip(x_test, y_test):
            output = self.forward_propagation(x, numHidLayer=numHidLayer, actFunc=actFunc, outActFunc=outActFunc)
            pred = np.argmax(output)
            predictions.append(pred == np.argmax(y))
        
        return np.mean(predictions)
    
    def fit(self, x_train, y_train):
        for x,y in zip(x_train, y_train):
            if (self.actFunc=="ReLu"):
                output = self.forward_propagation(x, numHidLayer=self.numHidLayer, actFunc=self.relu, outActFunc=self.softmax)
                changesOfWeights = self.backward_propagation(y, output, numHidLayer=self.numHidLayer, actFunc=self.relu, outActFunc=self.softmax)
                self.SGD(changesOfWeights)
            elif (self.actFunc=="sigmoid"):
                output = self.forward_propagation(x, numHidLayer=self.numHidLayer, actFunc=self.sigmoid, outActFunc=self.softmax)
                changesOfWeights = self.backward_propagation(y, output, numHidLayer=self.numHidLayer, actFunc=self.sigmoid, outActFunc=self.softmax)
                self.SGD(changesOfWeights)
            elif (self.actFunc=="tanh"):
                output = self.forward_propagation(x, numHidLayer=self.numHidLayer, actFunc=self.tanh, outActFunc=self.softmax)
                changesOfWeights = self.backward_propagation(y, output, numHidLayer=self.numHidLayer, actFunc=self.tanh, outActFunc=self.softmax)
                self.SGD(changesOfWeights)
                
            '''output = self.forward_propagation(x, numHidLayer=self.numHidLayer)
            changesOfWeights = self.backward_propagation(y, output, numHidLayer=self.numHidLayer)
            self.SGD(changesOfWeights)'''
    
    #val means test set
    def predict(self, x_test, y_test, start_time, iteration):
        #print("predicting.................")
        if (self.actFunc=="ReLu"):
            accuracy = self.evaluate_acc(x_test, y_test, numHidLayer=self.numHidLayer, actFunc=self.relu, outActFunc=self.softmax)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))
        elif(self.actFunc=="sigmoid"):
            accuracy = self.evaluate_acc(x_test, y_test, numHidLayer=self.numHidLayer, actFunc=self.sigmoid, outActFunc=self.softmax)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))
        elif(self.actFunc=="tanh"):
            accuracy = self.evaluate_acc(x_test, y_test, numHidLayer=self.numHidLayer, actFunc=self.tanh, outActFunc=self.softmax)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))
        return accuracy
        
        
        '''accuracy = self.evaluate_acc(x_test, y_test, numHidLayer=self.numHidLayer)
        print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
            iteration+1, time.time() - start_time, accuracy * 100
        ))'''
    
    def run(self, x_train, y_train, x_test, y_test):
        
        if(self.bias == True): # we already increased size when bias is true, therefore, we increase our dataset as well.
          x_train = np.hstack((x_train,np.ones((x_train.shape[0],1)))) #We add bias as 1 because of https://datascience.stackexchange.com/questions/11962/how-to-update-bias-and-biass-weight-using-backpropagation-algorithm
          x_test = np.hstack((x_test,np.ones((x_test.shape[0],1))))
        #print("@2:", x_train.shape, x_test.shape)
        accuracyArr = []
        start_time = time.time()
        for iteration in range(self.epochs):
            self.fit(x_train, y_train)
            accuracy = self.predict(x_test, y_test, start_time, iteration)
            accuracyArr.append(accuracy)
        return accuracyArr
            
    

    def train(self, x_train, y_train, x_test, y_test):
        start_time = time.time()
        for iteration in range(self.epochs):
            for x,y in zip(x_train, y_train):
                if (self.actFunc=="ReLu"):
                    output = self.forward_propagation(x, numHidLayer=self.numHidLayer, actFunc=self.relu, outActFunc=self.softmax)
                    changesOfWeights = self.backward_propagation(y, output, numHidLayer=self.numHidLayer, actFunc=self.relu, outActFunc=self.softmax)
                    self.SGD(changesOfWeights)
                elif (self.actFunc=="sigmoid"):
                    output = self.forward_propagation(x, numHidLayer=self.numHidLayer, actFunc=self.sigmoid, outActFunc=self.softmax)
                    changesOfWeights = self.backward_propagation(y, output, numHidLayer=self.numHidLayer, actFunc=self.sigmoid, outActFunc=self.softmax)
                    self.SGD(changesOfWeights)
                elif (self.actFunc=="tanh"):
                    output = self.forward_propagation(x, numHidLayer=self.numHidLayer, actFunc=self.tanh, outActFunc=self.softmax)
                    changesOfWeights = self.backward_propagation(y, output, numHidLayer=self.numHidLayer, actFunc=self.tanh, outActFunc=self.softmax)
                    self.SGD(changesOfWeights)
                    
                '''output = self.forward_propagation(x, numHidLayer=self.numHidLayer)
                changesOfWeights = self.backward_propagation(y, output, numHidLayer=self.numHidLayer)
                self.SGD(changesOfWeights)'''
            
            if (self.actFunc=="ReLu"):
                accuracy = self.evaluate_acc(x_test, y_test, numHidLayer=self.numHidLayer, actFunc=self.relu, outActFunc=self.softmax)
                print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                    iteration+1, time.time() - start_time, accuracy * 100
                ))
            elif(self.actFunc=="sigmoid"):
                accuracy = self.evaluate_acc(x_test, y_test, numHidLayer=self.numHidLayer, actFunc=self.sigmoid, outActFunc=self.softmax)
                print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                    iteration+1, time.time() - start_time, accuracy * 100
                ))
            elif(self.actFunc=="tanh"):
                accuracy = self.evaluate_acc(x_test, y_test, numHidLayer=self.numHidLayer, actFunc=self.tanh, outActFunc=self.softmax)
                print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                    iteration+1, time.time() - start_time, accuracy * 100
                ))
            
            
            '''accuracy = self.evaluate_acc(x_test, y_test, numHidLayer=self.numHidLayer)
            print('Epoch: {0}, Time Spent: {1:.2f}s, Accuracy: {2:.2f}%'.format(
                iteration+1, time.time() - start_time, accuracy * 100
            ))'''

def plotting(accuracyArr, numRuns, title):
    xdim = np.arange(1, numRuns+1)
    plt.plot(xdim, accuracyArr,  label='accuracy')
    plt.legend()
    plt.xlabel('epochs')
    plt.ylabel('accuracy')
    plt.title(title)
    plt.show()

'''
print("Testing: ", 0)
numRuns = 10
model = MLP(sizes=[784, 10], epochs=numRuns, numHidLayer=0, actFunc="ReLu", outActFunc="softmax", bias=True)
accuracyArr = model.run(x_train, y_train, x_test, y_test)
plotting(accuracyArr=accuracyArr, numRuns=numRuns, title="Layers:  [784, 10], output Function: Softmax")
'''

print("Question: 1.a_no hidden layer")
numRuns = 5
model = MLP(sizes=[784, 10], epochs=numRuns, numHidLayer=0, actFunc="ReLu", outActFunc="softmax", bias=False)
accNoHidden = model.run(x_train, y_train, x_test, y_test)
#plotting(accuracyArr=accuracyArr, numRuns=numRuns, title="Layers:  [784, 10], output Function: Softmax")

print("Question: 1.b_one hidden layer")
#numRuns = 5
model = MLP(sizes=[784, 128, 10], epochs=numRuns, numHidLayer=1, actFunc="ReLu", outActFunc="softmax", bias=False)
accOneHidden = model.run(x_train, y_train, x_test, y_test)
#plotting(accuracyArr=accuracyArr, numRuns=numRuns, title="Layers:  [784, 128, 10], Activation Function: ReLu, output Function: Softmax")

print("Question: 1.c_two hidden Layer")
#numRuns = 5
model = MLP(sizes=[784, 128, 128, 10], epochs=numRuns, numHidLayer=2, actFunc="ReLu", outActFunc="softmax", bias=False)
accTwoHidden = model.run(x_train, y_train, x_test, y_test)
#plotting(accuracyArr=accuracyArr, numRuns=numRuns, title="Layers:  [784, 128, 128, 10], Activation Function: ReLu, output Function: Softmax")


xdim = np.arange(1, numRuns+1)
plt.plot(xdim, accNoHidden , label='Layers:  [784, 10]')
plt.plot(xdim, accOneHidden ,  label='Layers:  [784, 128, 10]')
plt.plot(xdim, accTwoHidden ,  label='Layers:  [784, 128, 128, 10]')
plt.title("Question 1: compare accuracy across 0, 1, and 2 hidden layers")
plt.legend()

#-----------------------------------------------------------------------------------

accRelu = accTwoHidden
print("Question: 2.a_sigmoid")
#numRuns = 5
model = MLP(sizes=[784, 128, 128, 10], epochs=numRuns, numHidLayer=2, actFunc="sigmoid", outActFunc="softmax")
accSigmoid = model.run(x_train, y_train, x_test, y_test)
#plotting(accuracyArr=accuracyArr, numRuns=numRuns, title="Layers:  [784, 128, 128, 10], Activation Function: sigmoid, output Function: Softmax")

print("Question: 2.b_tanh")
#numRuns = 5
model = MLP(sizes=[784, 128, 128, 10], epochs=numRuns, numHidLayer=2, actFunc="tanh", outActFunc="softmax")
accTanh = model.run(x_train, y_train, x_test, y_test)
#plotting(accuracyArr=accuracyArr, numRuns=numRuns, title="Layers:  [784, 128, 128, 10], Activation Function: tanh, output Function: Softmax")


xdim = np.arange(1, numRuns+1)
plt.plot(xdim, accSigmoid ,label='Acc: sigmoid ')
plt.plot(xdim, accTanh ,  label='Acc: tanh ')
plt.plot(xdim, accRelu ,  label='Acc: ReLu ')
plt.title("Question 2: Relu vs Sigmoid vs Tanh")
plt.legend()

#---------------------------------------------------------------------------------


print("Question: 3_ L2 Regularization")
#Find the optimum 
theLamda = np.arange(1, 8, 1)
accLamda = []
for i,l in enumerate(theLamda):
    numRuns = 2
    model = MLP(sizes=[784, 128, 128, 10], epochs=numRuns, numHidLayer=2, actFunc="ReLu", outActFunc="softmax", L2Reg=True, lamda=10.**-l, bias=True)
    accReg = model.run(x_train, y_train, x_test, y_test)
    accLamda.append(accReg[len(accReg)-1])

xdim = np.arange(1, 8, 1)
plt.plot(xdim, accLamda,  label='accuracy')
plt.legend()
plt.xlabel('k: lamda=10^-k')
plt.ylabel('accuracy')
plt.title("Accuracy vs Lamda for regularization")
plt.show()

#--------------------------------------------------------------------------------

print("Question: 4_Unormilized data") #Unormilized data
(x_train_unorm, y_train_unorm), (x_test_unorm, y_test_unorm) = tf.keras.datasets.mnist.load_data()
# Rescale the images from [0,255] to the [0.0,1.0] range. (Normalize)
x_train_unorm, x_test_unorm = x_train_unorm[..., np.newaxis], x_test_unorm[..., np.newaxis]
# non normalizing will reduce accuracy, changing speed won't help it.

# Vectorization
x_train_unorm = x_train_unorm.reshape(x_train_unorm.shape[0],x_train_unorm.shape[1]*x_train_unorm.shape[2])
x_test_unorm = x_test_unorm.reshape(x_test_unorm.shape[0],x_test_unorm.shape[1]*x_test_unorm.shape[2])
# Change y_train_unorm and y_test_unorm
ytemp = np.zeros((y_train_unorm.shape[0], 10))
for i in range(y_train_unorm.shape[0]):
    ytemp[i, y_train_unorm[i]] = 1
y_train_unorm = ytemp

ytemp = np.zeros((y_test_unorm.shape[0], 10))
for i in range(y_test_unorm.shape[0]):
    ytemp[i, y_test_unorm[i]] = 1
y_test_unorm = ytemp

numRuns = 5
model = MLP(sizes=[784, 128, 128, 10], epochs=numRuns, numHidLayer=2, actFunc="ReLu", outActFunc="softmax")
accUnormilized = model.run(x_train_unorm, y_train_unorm, x_test_unorm, y_test_unorm)
#plotting(accuracyArr=accuracyArr, numRuns=numRuns, title="Unormalized data, 2 hidden layers, Relu activation")

accNormilized = accTwoHidden

xdim = np.arange(1, numRuns+1)
plt.plot(xdim, accUnormilized ,label='Acc: unormilized ')
plt.plot(xdim, accNormilized ,  label='Acc: normilized ')
plt.title("Question 4: normilized vs unormilized data")
plt.legend()


#---------------------------------------------------------------------------------

print("Question: 5_Test vs Train Acc")
numRuns = 200
model = MLP(sizes=[784, 128, 128, 10], epochs=numRuns, numHidLayer=2, actFunc="ReLu", outActFunc="softmax")
print("acc for training set")
accuracyArrTrain = model.run(x_train, y_train, x_train, y_train)
print("acc for test set")
accuracyArrTest = model.run(x_train, y_train, x_test, y_test)
plotting(accuracyArr=accuracyArrTest, numRuns=numRuns, title="2 hidden layers, Relu activation, compare Test and Train accuracy")
xdim = np.arange(1, numRuns+1)
plt.plot(xdim, accuracyArrTrain,  label='Train accuracy')
plt.plot(xdim, accuracyArrTest,  label='Test accuracy')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title("Test vs Train accuracies")
plt.show()

#-----------------------------------------------------------------------------------Open Ended

print("Open Ended Stuff###########################################################")

print("The effect of removing bias")
print("without bias")
numRuns = 100
model = MLP(sizes=[784, 10], epochs=numRuns, numHidLayer=0, actFunc="ReLu", outActFunc="softmax", bias=False)
accNonBias = model.run(x_train, y_train, x_test, y_test)
#plotting(accuracyArr=accuracyArr, numRuns=numRuns, title="Layers:  [784, 10], output Function: Softmax")

print("including bias")
#numRuns = 100
model = MLP(sizes=[784, 10], epochs=numRuns, numHidLayer=0, actFunc="ReLu", outActFunc="softmax", bias=True)
accBias = model.run(x_train, y_train, x_test, y_test)
#plotting(accuracyArr=accuracyArr, numRuns=numRuns, title="Layers:  [784, 10], output Function: Softmax")
xdim = np.arange(1, numRuns+1)
plt.plot(xdim, accNonBias,  label='no bias')
plt.plot(xdim, accBias,  label='bias defined')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title("bias vs no bias")
plt.show()
#----------------------------------------------------------------------------------------
print("3 hidden layer: 3X128-----------------------")
numRuns = 10
model = MLP(sizes=[784, 128, 128, 128, 10], epochs=numRuns, numHidLayer=3, actFunc="ReLu", outActFunc="softmax")
accuracyArrNonNarrow = model.run(x_train, y_train, x_test, y_test)
#plotting(accuracyArr=accuracyArr, numRuns=numRuns, title="Layers:  [784, 128,128,128, 10], Activation Function: ReLu, output Function: Softmax")

print("3 hidden layer: 128=>64=>32")
numRuns = 10
model = MLP(sizes=[784, 128, 64, 32, 10], epochs=numRuns, numHidLayer=3, actFunc="ReLu", outActFunc="softmax")
accuracyArrNarrowing = model.run(x_train, y_train, x_test, y_test)
#plotting(accuracyArr=accuracyArr, numRuns=numRuns, title="Layers:  [784, 128, 64, 32, 10], Activation Function: ReLu, output Function: Softmax")
xdim = np.arange(1, numRuns+1)
plt.plot(xdim, accuracyArrNonNarrow,  label='equal layers')
plt.plot(xdim, accuracyArrNarrowing,  label='Narrowing layers')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title("Narrwoing vs equal layers")
plt.show()

#------------------------------------------------------------------------------------------
print("Testing: shuffling data---------------")
import sklearn.utils
x_train_shuffle, y_train_shuffle = sklearn.utils.shuffle(x_train,y_train)
x_test_shuffle, y_test_shuffle = sklearn.utils.shuffle(x_test,y_test)
numRuns = 200
model = MLP(sizes=[784, 128, 128, 10], epochs=numRuns, numHidLayer=2, actFunc="ReLu", outActFunc="softmax")
accShuffled = model.run(x_train, y_train, x_test, y_test)
#plotting(accuracyArr=accuracyArr, numRuns=numRuns, title="Layers:  [784, 10], Shuffled")
accNonshuffeled = accTwoHidden
xdim = np.arange(1, numRuns+1)
plt.plot(xdim, accShuffled,  label='shuffled data')
plt.plot(xdim, accNonshuffeled,  label='normal data')
plt.legend()
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title("shuffled vs non shuffled data")
plt.show()


#--------------------------------------------------------------------------------------------
print("Training with differnt number of inputs----------")
numRuns = 10
ydim=[]
for i in range(0,5):
    tail = int(10**i)
    print("Number of input = ", tail)
    model = MLP(sizes=[784, 128, 128, 10], epochs=numRuns, numHidLayer=2, actFunc="ReLu", outActFunc="softmax")
    accuracyArr = model.run(x_train[0:tail], y_train[0:tail], x_test[0:tail], y_test[0:tail])
    ydim.append(accuracyArr[len(accuracyArr)-1])

xdim = [1, 10, 100, 1000, 10000]                            
plt.plot(xdim, ydim,  label='Accuracy')
plt.legend()
plt.xlabel('number of input data')
plt.ylabel('accuracy')
plt.title("accuracy vs number of inputs")
plt.show()
 
#-------------------------------------------------------------------------------------------

print("Training with differnt learning rates----------")
numRuns = 4
ydim=[]
for i in range(0,5):
    l_rate = 10**-i #1, 0.1, 0.01, 0.001
    print("learning rate = ", l_rate)
    model = MLP(sizes=[784, 128, 128, 10], epochs=numRuns, learningRate=l_rate,numHidLayer=2, actFunc="ReLu", outActFunc="softmax")
    accuracyArr = model.run(x_train, y_train, x_test, y_test)
    ydim.append(accuracyArr[len(accuracyArr)-1])

xdim = [1, 0.1, 0.01, 0.001, 0.0001]                            
plt.plot(xdim, ydim,  label='Accuracy')
plt.legend()
plt.xlabel('learning rate')
plt.ylabel('accuracy')
plt.title("accuracy vs learning rate")
plt.show()


####################augmentation with shift##################
#For augmentation part, we have inspired from the following link
#https://towardsdatascience.com/improving-accuracy-on-mnist-using-data-augmentation-b5c38eb5a903
from scipy.ndimage.interpolation import shift
from scipy.ndimage.interpolation import rotate
###Data Augmentation###
def shift_image(image, x, y):
    image = image.reshape((28, 28))
    shifted_image = shift(image, [y, x], cval=0, mode="constant")
    return shifted_image.reshape([-1])
#################

# Creating Augmented Dataset with shift
X_train_augmented = [image for image in x_train]
y_train_augmented = [image for image in y_train]
#we shift pictures in four directions
for dx, dy in ((30,0) , (0,0)):
     for image, label in zip(x_train, y_train):
             X_train_augmented.append(shift_image(image, dx, dy))
             y_train_augmented.append(label)

# Shuffle the dataset
shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = np.array(X_train_augmented)[shuffle_idx]
y_train_augmented = np.array(y_train_augmented)[shuffle_idx]
##########
numRuns = 10

#####2 hiddem layers in shifted data
print("Augmented data with 2 layers shifted data")
model = MLP(sizes=[784, 128, 128, 10], epochs=numRuns, numHidLayer=2, actFunc="ReLu", outActFunc="softmax", bias=True)
accuracyArrTwoHAug = model.run(X_train_augmented, y_train_augmented, x_test, y_test)
#########2 hidden layers in normal data
print("Normal data with 2 layers")
model = MLP(sizes=[784, 128, 128, 10], epochs=numRuns, numHidLayer=2, actFunc="ReLu", outActFunc="softmax", bias=True)
accuracyArrTwoNor = model.run(x_train, y_train, x_test, y_test)

####################augmentation with rotate part#####################
from scipy import ndimage, misc
# Creating Augmented Dataset
X_train_augmented = [image for image in x_train]
y_train_augmented = [image for image in y_train]
#we shift pictures in four directions
#for degree in angles:

def rotate_image(image, degree):
    image = image.reshape((28, 28))
    rotated_image = rotate(image, degree, cval=0, mode="constant",reshape=False)
    #print(rotated_image.shape)
    return rotated_image.reshape([-1])
angles = [5]
for degree in angles:
  for image, label in zip(x_train, y_train):
             X_train_augmented.append(rotate_image(image, degree))
             y_train_augmented.append(label)

# Shuffle the dataset
shuffle_idx = np.random.permutation(len(X_train_augmented))
X_train_augmented = np.array(X_train_augmented)[shuffle_idx]
y_train_augmented = np.array(y_train_augmented)[shuffle_idx]
print("Augmented data with 2 layers rotated data")
model = MLP(sizes=[784, 128, 128, 10], epochs=numRuns, numHidLayer=2, actFunc="ReLu", outActFunc="softmax", bias=True)
accuracyArrTwoHAugRot = model.run(X_train_augmented, y_train_augmented, x_test, y_test)
###plot###
import matplotlib.pyplot as plt
xdim = np.arange(1, numRuns+1)
plt.xlabel('epochs')
plt.ylabel('accuracy')
plt.title('accuracy comparison')
plt.plot(xdim, accuracyArrTwoNor ,label='Normal data two layers')
plt.plot(xdim, accuracyArrTwoHAug ,label='Augmented shift two layers')
plt.plot(xdim, accuracyArrTwoHAugRot ,label='Augmented rotate two layers')
plt.legend()
