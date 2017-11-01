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
import copy

seed = 3
np.random.seed(seed)

dire = '../environment/out/' 

X_train = np.loadtxt(dire+'X_train.txt')
Y_train = np.loadtxt(dire+'Y_train.txt')

X_test = np.loadtxt(dire+'X_val.txt')
Y_test = np.loadtxt(dire+'Y_val.txt')

# adjust to ONLY predict sofa
sofa_index = 29
Y_train = Y_train[:, 29]
Y_train = Y_train[:, None]

Y_test = Y_test[:,29]
Y_test = Y_test[:, None]



#mean = [0,0,0]
#cov = [[1,0,0], [0,10,0], [0,0,1]]
#X_train = np.random.multivariate_normal(mean, cov, size=1000)
#print X_train.shape
#import copy
#tmp = 5+5*copy.deepcopy(X_train)
#Y_train = np.sum(tmp, axis = 1)
#Y_train = Y_train[:, None]

#np.savetxt(dire + 'X_train_expt.txt',X_train,fmt='%5.4f')
#np.savetxt(dire + 'Y_train_expt.txt',Y_train,fmt='%5.4f')

for i in zip(X_train[:5], Y_train[:5]):
    print i

state_dim = X_train.shape[1]
tar_dim = Y_train.shape[1]

print "state_dim is ", state_dim
print "target_dim is ", tar_dim

params_task = {} 
params_task['seed'] = seed
params_task['state_dim'] = X_train.shape[1]
params_task['r_dim'] = Y_train.shape[1]


params_model = {}
params_model['mode'] = 'AD_sofa'
params_model['saved'] = False
params_model['epochs'] = 1200
params_model['batchsize'] = int(X_train.shape[0]/300)
params_model['alpha'] = float(sys.argv[1])
params_model['learn_rate'] = np.float32(0.001)
params_model['samples'] =  50

params_model['dimh'] = 300
params_model['graph'] = [params_task['state_dim'] ,params_model['dimh'],params_model['dimh'],params_task['r_dim']]

X = X_train
Y = Y_train

model = AD(params_model,params_task,X,Y, xtest=X_test, ytest=Y_test)
#model = AD(params_model,params_task,X,Y, xtest=X, ytest=Y)
model.train() 

saved = {}
saved['params_model'] = params_model
saved['params_task'] = params_task
saved['model_norm'] = [model.mean_X,model.std_X,model.mean_Y,model.std_Y]
saved['model_weights'] = model.get_weights() 

dire = 'models/'
outstr = params_model['mode'] + '_' + str(params_model['alpha'])  + '.p'
pickle.dump(saved,open(dire + outstr,'wb'))

