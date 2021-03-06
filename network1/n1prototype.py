# -*- coding: utf-8 -*-
"""
Created on Wed Aug 12 09:49:47 2015

@author: jbarnett

This is a prototype for a neural network that will identify tomato images.
It requires two directories, one consisting exclusively of .png tomato images
the other consisting exclusively of .png non-tomato images
the idea is to read both of these in and append them together using resizetoInputArray
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
from PIL import Image



script_dir = os.path.dirname(__file__)
positivepath = "data/tomatoGeneral/*.png" # due to glob mechanics, *.png must be added, filters to only png images
negativepath = "data/nonTomatoGeneral/*.jpg"
testpath = "data/testImages/*.png"


def preprocess_images(path):
    '''
    resizes and converts all images in files to rgb format and returns list containing pixel values for all images
    :param path: dir where files are stored
    :return: a list of arrays containing all pixels in the image in format ([r,g,b,...],[r,g,b,...],...)
    '''

    resArray = []
    abs_file_path = os.path.join(script_dir, path) # add subdirectory to our global path
    files = glob.glob(abs_file_path) # use Glob to get a whole list of the images
    for item in files: 
        im = Image.open(item).resize((50, 50)).convert('RGB')    # read it in
        resized = np.array([rgb for pixel in list(im.getdata()) for rgb in pixel])  # converts 2500 (r,g,b) into 7500
        resArray.append(resized)
    return resArray


class TomatoIdentifier(DenseDesignMatrix):
    def __init__(self):
        X = []
        Y = []
        self.class_names = ['0', '1']
        X = preprocess_images(positivepath)
        temp = len(X)
        i = 0
        while i < temp:
            Y.append((1, 0))
            i += 1
        X += preprocess_images(negativepath)
        temp2 = len(X)-temp
        for i in range(temp2):
            Y.append((0,1))#need to fill the outputs, repeat (1,0) for size of positivepath, () for size of negative
        X = np.array(X)
        Y = np.array(Y)
        super(TomatoIdentifier, self).__init__(X=X, y=Y)

# create dataset
ds = TomatoIdentifier()
# create hidden layer with 50 nodes, init weights in range -0.1 to 0.1 and add
# a bias with value 1
hidden_layer = mlp.Sigmoid(layer_name='hidden', dim=80, irange=.2, init_bias=1.)
# create Softmax output layer
output_layer = mlp.Softmax(2, 'output', irange=.1)
# create Stochastic Gradient Descent trainer that runs for 500 epochs
trainer = sgd.SGD(learning_rate=.05, batch_size=10, termination_criterion=EpochCounter(100))
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

#add and process test images    
TestImages = []
TestImages = preprocess_images(testpath)

#test them
inputs = np.array(TestImages)
print ann.fprop(theano.shared(inputs, name='inputs')).eval()
#inputs = TestImages[1]
#print ann.fprop(theano.shared(inputs, name='inputs')).eval()
#inputs = TestImages[2]
#print ann.fprop(theano.shared(inputs, name='inputs')).eval()
#inputs = TestImages[3]
#print ann.fprop(theano.shared(inputs, name='inputs')).eval()
