#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Jan 30 17:53:43 2019

@author: adrianp
"""
import numpy as np

class Network(object):
    def __init__(self,sizes):
        self.sizes = sizes
        self.layers = len(sizes)
        self.hidden_layers = self.layers-2
        self.input_size = sizes[0]
        self.output_size = sizes[-1]
        self.biases = [np.random.randn(y) for y in sizes[1:]]
        self.weights = [np.random.randn(x,y) for x,y in zip(sizes[1:],sizes[:-1])]
        
    def feedforward(self,act):
        if len(act)!=self.weights[0].shape[1]: 
            raise ValueError('The length of the activation data is not correct, it should be %s' % self.weights[0].shape[1]) 
        act=np.asfarray(act)
        for w,b in zip(self.weights,self.biases):
            act = sigmoid(w@act + b)
            
        return act
        
    def evaluate(self,test_data):
        return sum(int(np.argmax(self.feedforward(x)) == y) for x,y in test_data)
    
    def back_prop(self,act,sol):
        acts = [act]
        zs = []
        D_b = [np.zeros(b.shape) for b in self.biases]
        D_w = [np.zeros(w.shape) for w in self.weights]
        
        for w,b in zip(self.weights,self.biases):
            z = w@act + b
            zs.append(z)
            act = sigmoid(z)
            acts.append(act)

        delta = (der_cost(acts[-1],sol)*der_sigmoid([zs[-1]])).flatten()
        D_b[-1] = delta
        D_w[-1] = np.outer(delta,acts[-2])
        for l in range(2,self.layers):
            delta = ((self.weights[-l + 1].transpose()@delta)*der_sigmoid(zs[-l])).flatten()
            D_b[-l] = delta
            D_w[-l] = np.outer(delta,acts[-l-1])
           
        return D_b,D_w
        
    def training_session(self,data,eta):
        nab_b = [np.zeros(b.shape) for b in self.biases]
        nab_w = [np.zeros(w.shape) for w in self.weights]
        for x,y in data:
            D_b,D_w = self.back_prop(x,y)
            nab_b = [n_b + d_b for n_b,d_b in zip(nab_b, D_b)]
            nab_w = [n_w + d_w for n_w,d_w in zip(nab_w, D_w)]
        
        self.weights = [w - eta*n_w/len(data) for w,n_w in zip(self.weights,nab_w)]
        self.biases = [b - eta*n_b/len(data) for b,n_b in zip(self.biases,nab_b)]
        
        
    def SGD(self, train_data, train_size, workouts, eta = 3,test_data = None):
        if test_data: n_test = len(test_data)
        n_train = len(train_data)
        
        for time in range(workouts):
            trains = [train_data[k: k + train_size] for k in range(0,n_train,train_size)]
            for train in trains:
                self.training_session(train,eta)
                
            if test_data:
                print('Workout {} done: {}/{}'.format(time,self.evaluate(test_data),n_test))
            else:
                print('Workout {} done'.format(time))
    
def sigmoid(z):
    z=np.asfarray(z)
    return 1/(1+np.exp(-z))

def der_sigmoid(z):
    z=np.asfarray(z)
    return sigmoid(z)*(1-sigmoid(z))

def der_cost(out,sol):
    y = np.zeros(len(out))
    y[sol]=1;
    return out-y
