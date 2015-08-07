# NOTE!!!!
# This code is a very slight modification of code by Gustav Arngarden
#original code can be found at http://www.arngarden.com/2013/07/29/neural-network-example-using-pylearn2/
# I basically just changed the dimensionality and switched it from a simple XOR example to XOR-and-XOR
#compile using python xorandxornetwork.py develop

import theano
from pylearn2.models import mlp
from pylearn2.training_algorithms import sgd
from pylearn2.termination_criteria import EpochCounter
from pylearn2.datasets.dense_design_matrix import DenseDesignMatrix
import numpy as np
from random import randint

class XORandXOR(DenseDesignMatrix):
    def __init__(self):
        self.class_names = ['0', '1']
        X = [[randint(0, 1), randint(0, 1),randint(0,1),randint(0,1)] for _ in range(1000)]
        y = []
        for a, b, c, d in X:
            if a + b == 1:
                if c + d == 1:
                    y.append([0, 1])
                else:
                    y.append([1,0])
            else:
                y.append([1, 0])
        X = np.array(X)
        y = np.array(y)
        super(XORandXOR, self).__init__(X=X, y=y)

# create XOR dataset
ds = XORandXOR()
# create hidden layer with 4  nodes, init weights in range -0.1 to 0.1 and add
# a bias with value 1
hidden_layer = mlp.Sigmoid(layer_name='hidden', dim=4, irange=.1, init_bias=1.)
# create Softmax output layer
output_layer = mlp.Softmax(2, 'output', irange=.1)
# create Stochastic Gradient Descent trainer that runs for 400 epochs
trainer = sgd.SGD(learning_rate=.05, batch_size=10, termination_criterion=EpochCounter(400))
layers = [hidden_layer, output_layer]
# create neural net that takes four inputs
ann = mlp.MLP(layers, nvis=4)
trainer.setup(ann, ds)
# train neural net until the termination criterion is true
while True:
    trainer.train(dataset=ds)
    ann.monitor.report_epoch()
    ann.monitor()
    if not trainer.continue_learning(ann):
        break

inputs = np.array([[0, 0, 0, 0]])
print ann.fprop(theano.shared(inputs, name='inputs')).eval()
inputs = np.array([[0, 1, 0, 1]])
print ann.fprop(theano.shared(inputs, name='inputs')).eval()
inputs = np.array([[1, 0, 1, 1]])
print ann.fprop(theano.shared(inputs, name='inputs')).eval()
inputs = np.array([[1, 1, 0, 1]])
print ann.fprop(theano.shared(inputs, name='inputs')).eval()
