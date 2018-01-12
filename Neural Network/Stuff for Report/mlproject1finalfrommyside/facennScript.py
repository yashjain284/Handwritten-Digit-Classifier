'''
Comparing single layer MLP with deep MLP (using TensorFlow)
'''

import numpy as np
from scipy.optimize import minimize
from math import sqrt
import pickle
import time

# Do not change this
def initializeWeights(n_in,n_out):
    """
    # initializeWeights return the random weights for Neural Network given the
    # number of node in the input layer and output layer

    # Input:
    # n_in: number of nodes of the input layer
    # n_out: number of nodes of the output layer
                            
    # Output: 
    # W: matrix of random initial weights with size (n_out x (n_in + 1))"""
    epsilon = sqrt(6) / sqrt(n_in + n_out + 1);
    W = (np.random.rand(n_out, n_in + 1)*2* epsilon) - epsilon;
    return W



# Replace this with your sigmoid implementation
def sigmoid(z):
    return 1.0/(1.0+np.exp(-z));

def nnObjFunction(params, *args):

    n_input, n_hidden, n_class, training_data, training_label, lambdaval = args

    w1 = params[0:n_hidden * (n_input + 1)].reshape((n_hidden, (n_input + 1)))
    w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))
    obj_val = 0
    # Your code here
    x = training_data
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    w1 = w1.transpose()
    aj = np.dot(x, w1)
    zj = sigmoid(aj)
    zj = np.hstack((zj, np.ones((zj.shape[0], 1))))
    w2 = w2.transpose()
    bl = np.dot(zj, w2)
    ol = sigmoid(bl)

    n = len(training_data)
    y = np.zeros(shape=(n, n_class))
    training_label = training_label.astype(np.int)
    for i in range(0, n):
        y[i, training_label[i]] = 1

    jw = -(np.sum(y * np.log(ol) + (1 - y) * np.log(1 - ol)))/n
    w1 = w1.transpose()
    w2 = w2.transpose()
    sumw1 = 0.0
    sumw2 = 0.0
    for j in range(0, n_hidden):
        for p in range(0, n_input + 1):
            sumw1 = sumw1 + w1[j][p] * w1[j][p]

    for l in range(0, n_class):
        for j in range(0, n_hidden + 1):
            sumw2 = sumw2 + w2[l][j] * w2[l][j]

    obj_val = jw + ((lambdaval /(2 * n)) * (np.sum(np.square(w1)) + np.sum(np.square(w2))))
    #print("obj_val = ",obj_val)

    deltal = ol - y
    deltaltrp = deltal.transpose()
    dw2lj = np.dot(deltaltrp, zj)
    grad_w2 = (dw2lj + lambdaval * w2)/n
    # print("grad_w2 = ",grad_w2.shape)

    temp = (zj * (1 - zj) * np.dot(deltal, w2)).transpose()
    tempvar = np.dot(temp, x);
    tempvar = np.delete(tempvar, len(tempvar) - 1, 0)
    grad_w1 = (tempvar + lambdaval * w1)/n
    # print("grad_w1 = ", grad_w1.shape)


    # Make sure you reshape the gradient matrices to a 1D array. for instance if your gradient matrices are grad_w1 and grad_w2
    # you would use code similar to the one below to create a flat array
    obj_grad = np.concatenate((grad_w1.flatten(), grad_w2.flatten()), 0)
    # obj_grad = np.array([])

    return (obj_val, obj_grad)


def nnPredict(w1, w2, data):
    x = data
    x = np.hstack((x, np.ones((x.shape[0], 1))))
    w1 = w1.transpose()
    aj = np.dot(x, w1);
    zj = sigmoid(aj);
    zj = np.hstack((zj, np.ones((zj.shape[0], 1))))
    w2 = w2.transpose()
    bl = np.dot(zj, w2);
    ol = sigmoid(bl);
    labels = np.argmax(ol, axis=1)
    return labels

# Do not change this
def preprocess():
    pickle_obj = pickle.load(file=open('face_all.pickle', 'rb'))
    features = pickle_obj['Features']
    labels = pickle_obj['Labels']
    train_x = features[0:21100] / 255
    valid_x = features[21100:23765] / 255
    test_x = features[23765:] / 255

    labels = labels[0]
    train_y = labels[0:21100]
    valid_y = labels[21100:23765]
    test_y = labels[23765:]
    return train_x, train_y, valid_x, valid_y, test_x, test_y

"""**************Neural Network Script Starts here********************************"""
train_data, train_label, validation_data, validation_label, test_data, test_label = preprocess()
#  Train Neural Network
# set the number of nodes in input unit (not including bias unit)
n_input = train_data.shape[1]
# set the number of nodes in hidden unit (not including bias unit)
n_hidden = 256
# set the number of nodes in output unit
n_class = 2

# initialize the weights into some random matrices
initial_w1 = initializeWeights(n_input, n_hidden);
initial_w2 = initializeWeights(n_hidden, n_class);
# unroll 2 weight matrices into single column vector
initialWeights = np.concatenate((initial_w1.flatten(), initial_w2.flatten()),0)
# set the regularization hyper-parameter
lambdaval = 10;
args = (n_input, n_hidden, n_class, train_data, train_label, lambdaval)

#Train Neural Network using fmin_cg or minimize from scipy,optimize module. Check documentation for a working example
opts = {'maxiter' :50}    # Preferred value.
starttime = time.time()
nn_params = minimize(nnObjFunction, initialWeights, jac=True, args=args,method='CG', options=opts)
endtime = time.time()
print('time to train = ',endtime-starttime)
params = nn_params.get('x')
#Reshape nnParams from 1D vector into w1 and w2 matrices
w1 = params[0:n_hidden * (n_input + 1)].reshape( (n_hidden, (n_input + 1)))
w2 = params[(n_hidden * (n_input + 1)):].reshape((n_class, (n_hidden + 1)))

#Test the computed parameters
predicted_label = nnPredict(w1,w2,train_data)
#find the accuracy on Training Dataset
print('\n Training set Accuracy:' + str(100*np.mean((predicted_label == train_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,validation_data)
#find the accuracy on Validation Dataset
print('\n Validation set Accuracy:' + str(100*np.mean((predicted_label == validation_label).astype(float))) + '%')
predicted_label = nnPredict(w1,w2,test_data)
#find the accuracy on Validation Dataset
print('\n Test set Accuracy:' +  str(100*np.mean((predicted_label == test_label).astype(float))) + '%')