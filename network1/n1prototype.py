# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:49:47 2015

@author: jbarnett

This is a prototype for a neural network that will identify tomato images.
It requires two directories, one consisting exclusively of .png tomato images
the other consisting exclusively of .png non-tomato images
the idea is to read both of these in and append them togeth using resizetoInputArray
then create a corresponding output array based on the length of these, with (0,1) indicating a positive result
and (1,0) indicating a negative result. (We could switch this, doesn't really matter lol)

The problems currently with it are that resizetoInputArray is not giving us raw numbers, but a tuple of ([numbersweneed],encoding).
We got to extract [numbersweneed] from each of these, and create array([numbers1],[numbers2], ...)

Have not tested the output length generation stuff yet.

Want to work on further abstracting this code to work with an arbitrary number of parameters/directories
"""

import theano
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np
from random import randint
import os
from scipy import misc
import glob




script_dir = os.path.dirname(__file__)
positivepath = "data/tomatoGeneral/*.png" # due to glob mechanics, *.png must be added, filters to only png images
positivepath = "data/nonTomatoGeneral/*.jpg" 


def resizetoInputArray(resArray,path): #takes in list to store it, as well as a path
    abs_file_path = os.path.join(script_dir, path) #add subdirectory to our global path
    files=glob.glob(abs_file_path) #use Glob to get a whole list of the images
    for item in files: 
        im = misc.imread(item) #read it in
        print(len(im))
        imResize = misc.imresize(im,(50,50)) #resize it to 50x50
        print(len(imResize))
        imFlattened = imResize.flatten() #flatten that shit        
        print(len(imFlattened)) #should be 7500, a lot are 10000? #also, need to get rid of that weird encoding tuple stuff
        #that tuple encoding stuff is causing a vectorspace error in the densedesignmatrix that tomatoidentifier stems from
        resArray.append(imFlattened)
    return resArray	


class TomatoIdentifier(DenseDesignMatrix):
    def __init__(self):
        X = []
        Y = []
        self.class_names = ['0', '1']
        X = resizetoInputArray(X,positivepath) 
        temp = len(X)
        i = 0
        while i < temp:
            Y.append((1,0))
            i = i + 1
        #X += resizetoInputArray(X,negativepath) 
        #temp2 = len(X)-temp
        #for i in temp2:
        #    Y.append((0,1))#need to fill the outputs, repeat (1,0) for size of positivepath, () for size of negative
        X = np.array(X)
        Y = np.array(Y)
        super(TomatoIdentifier, self).__init__(X=X, y=Y)

# create dataset
ds = TomatoIdentifier()
# create hidden layer with 50 nodes, init weights in range -0.1 to 0.1 and add
# a bias with value 1
hidden_layer = mlp.Sigmoid(layer_name='hidden', dim=50, irange=.1, init_bias=1.)
# create Softmax output layer
output_layer = mlp.Softmax(2, 'output', irange=.1)
# create Stochastic Gradient Descent trainer that runs for 500 epochs
trainer = sgd.SGD(learning_rate=.05, batch_size=10, termination_criterion=EpochCounter(500))
layers = [hidden_layer, output_layer]
# create neural net that takes two inputs
ann = mlp.MLP(layers, nvis=7500)
trainer.setup(ann, ds)
# train neural net until the termination criterion is true
while True:
    trainer.train(dataset=ds)
    ann.monitor.report_epoch()
    ann.monitor()
    if not trainer.continue_learning(ann):
        break

inputs = np.array([[0, 0]])
print ann.fprop(theano.shared(inputs, name='inputs')).eval()
inputs = np.array([[0, 1]])
print ann.fprop(theano.shared(inputs, name='inputs')).eval()
inputs = np.array([[1, 0]])
print ann.fprop(theano.shared(inputs, name='inputs')).eval()
inputs = np.array([[1, 1]])
print ann.fprop(theano.shared(inputs, name='inputs')).eval()
