# coding=utf-8
from __future__ import division
'''
The MIT License (MIT)

Copyright 2017 Siemens AG, University of Cambridge

Authors: Stefan Depeweg, José Miguel Hernández-Lobato

Permission is hereby granted, free of charge, to any person obtaining a copy
of this software and associated documentation files (the "Software"), to deal
in the Software without restriction, including without limitation the rights
to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
copies of the Software, and to permit persons to whom the Software is
furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all
copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
SOFTWARE.
'''
import numpy as np
import sys
sys.path.append('../')

from models.AD import AD
import time

import os
import pdb
import pylab as plt
import pickle
import math
import theano
import theano.tensor as T
import lasagne

seed = 3
np.random.seed(seed)

params_task = {}
params_task['task'] = 'sepsis_controller'
params_task['seed'] = seed

dire = '../environment/out/' 

X_train = np.loadtxt(dire+'X_train.txt')
Y_train = np.loadtxt(dire+'Y_train.txt')
X_test = np.loadtxt(dire+'X_val.txt')
Y_test = np.loadtxt(dire+'Y_val.txt')

# load in the saved factors for the rewards
sofa_mean = np.load('../../data/sofa_mean.npy')
sofa_std = np.load('../../data/sofa_std.npy')
lact_mean = np.load('../../data/lact_mean.npy')
lact_std = np.load('../../data/lact_std.npy')

# load in the state vector so we know what indices SOFA and lactate are stored at
state_vec = list(np.loadtxt('../../data/state_features.txt', dtype=str))

sofa_index = state_vec.index('SOFA')
lact_index = state_vec.index('Arterial_lactate')

# Let X_start be the state vector used by the policy network
# X_train is a tuple of (states, actions)
# We want X_start to be just the states -- this is what we use to decide how to act
X_start = X_train[:, :-2]

params_task['state_dim'] = X_start.shape[1]
params_task['action_dim'] = 2

print params_task['state_dim']

dire = 'models/'
alpha = sys.argv[1]
fstr = 'AD_'+alpha+ '.p'
params_model = pickle.load(open(dire+fstr,'rb'))['params_model']
params_model['saved'] = dire+fstr

# load the model
print params_model['saved']
model = AD(params_model,params_task,X_train,Y_train)
model.loadWeights() 


# define the policy search parameters
params_controller = {}
params_controller['saved'] =  False
params_controller['learning_rate'] = 0.0001
params_controller['name'] = 'controller'
params_controller['T'] = 10
params_controller['epochs'] = 300
params_controller['batchsize'] = 25
params_controller['samples'] = 25

# add some params to unscale rewards
params_controller['sofa_mean'] = sofa_mean
params_controller['sofa_std'] = sofa_std
params_controller['lact_mean'] = lact_mean
params_controller['lact_std'] = lact_std

# and where to find these fields in the state vector
params_controller['sofa_index'] = sofa_index
params_controller['lact_index'] = lact_index

def policy(input_var=None):
    l_in = lasagne.layers.InputLayer(shape=(None, X_start.shape[1]),input_var=input_var)
    l_hid1 = lasagne.layers.DenseLayer(l_in, num_units=128,nonlinearity=lasagne.nonlinearities.rectify)
    l_hid2 = lasagne.layers.DenseLayer(l_hid1, num_units=64,nonlinearity=lasagne.nonlinearities.rectify)
    l_out = lasagne.layers.DenseLayer(l_hid2,num_units=2,nonlinearity=lasagne.nonlinearities.tanh)
    return l_out


import controller.PolicySearch
contrl = controller.PolicySearch.PolicySearch(params_controller,params_task,X_train,model,policy())

errs = []
trace = []

from timeit import default_timer as timer
start = timer()
time_epochs = []

model.bb_alpha.network.update_randomness(params_controller['samples'])
for j in range(params_controller['epochs']):
    errs = []
    inds = np.random.permutation(X_train.shape[0])
    for k in range(100):
        ind = inds[k*params_controller['batchsize']:(k+1)*params_controller['batchsize']]
        model.bb_alpha.network.update_randomness(params_controller['samples'])
        e = contrl.train_func(X_train[ind])
        errs.append(e)

    end = timer()
    time_e = end-start
    time_epochs.append(time_e)
    atime_e = np.mean(time_epochs[-5:])
    rest_time = int(atime_e * (params_controller['epochs'] - (j+1)))
    rest_hours,rest_seconds = divmod(rest_time,60*60)
    rest_minutes,_ = divmod(rest_seconds,60)
    err = np.mean(errs)
    print 'Remaining: ' + str(rest_hours) + 'h:'  + str(rest_minutes) +  'm,  Policy Cost: ' + str(err) 
    trace.append(err)
    start = timer()


weights = []
for p in lasagne.layers.get_all_params(contrl.policy,trainable=True):
    weights.append(p.get_value())

saved = {}
saved['params_model'] = params_model
saved['params_controller'] = params_controller
saved['params_task'] = params_task
saved['model_norm'] = [model.mean_X,model.std_X,model.mean_Y,model.std_Y]
saved['trace'] = trace
saved['controller_weights'] = weights
dire = 'controller/'
pickle.dump(saved,open(dire + fstr,'wb'))
