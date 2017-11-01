# coding=utf-8
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
import cPickle as pickle
import theano
import theano.tensor as T
from  theano import shared
import pdb
from collections import OrderedDict
import math
from bb_alpha_z.black_box_alpha import BB_alpha as bbaz

class AD:
    def __init__(self,params,params_task,X,Y,xtest=None, ytest=None):
        """ Wrapper class for the bb-\alpha code. 
        Implements normalization, loading/retrieving weights,
        and a prediction method
        """

        self.params = params
        self.params_task = params_task
        self.X,self.Y = self.normalize(X,Y)

        n_train = self.X.shape[0]
        self.X = T.cast(theano.shared(self.X), theano.config.floatX)
        self.Y = T.cast(theano.shared(self.Y), theano.config.floatX)
        
        if xtest is None:
            self.X_test = None
            n_test = 0
        else:
            self.X_test = (xtest-self.mean_X) / self.std_X
            n_test = self.X_test.shape[0]
            self.X_test = T.cast(theano.shared(self.X_test), theano.config.floatX)

            
        if ytest is None:
            self.Y_test = None
        else:
            self.Y_test = (ytest-self.mean_Y) / self.std_Y
            self.Y_test = T.cast(theano.shared(self.Y_test), theano.config.floatX)
         

        self.bb_alpha = bbaz(self.params['graph'], self.params['samples'], self.params['alpha'], self.params['learn_rate'],  1.0, self.params['batchsize'], self.X,self.Y,n_train, self.X_test, self.Y_test, n_test)

    def normalize(self,X,Y):
        self.std_X = np.std(X,axis=0)
        self.std_X[self.std_X == 0] = 1.
        self.mean_X = np.mean(X,axis=0)

        self.std_Y = np.std(Y,axis=0)
        self.std_Y[self.std_Y == 0] = 1.
        self.mean_Y = np.mean(Y,axis=0)
        X_norm = (X - self.mean_X) / self.std_X
        Y_norm = (Y - self.mean_Y) / self.std_Y
        return (X_norm,Y_norm)


    def loadWeights(self):
        if self.params['saved'] != False:

            dat = pickle.load(open(self.params['saved'],'rb')) 
            weights = dat['model_weights']
            for j,w in enumerate(weights):
                p = self.bb_alpha.network.params[j]
                
                # q(z|(x,y)) is only used during for training
                if p.name != 'log_v_z_par' and p.name != 'm_z_par':
                    p.set_value(np.array(w,dtype=theano.config.floatX))

            if dat.has_key('model_norm'):
                self.mean_X = dat['model_norm'][0]
                self.std_X = dat['model_norm'][1]
                self.mean_Y = dat['model_norm'][2]
                self.std_Y = dat['model_norm'][3]
            return


    def train(self,epochs=0):
        if epochs == 0:
            self.bb_alpha.train(self.params['epochs'])
        else:
            self.bb_alpha.train(epochs)


    def  predict(self,X_test, mode='numerical',provide_noise=False,noise=None ):
        """ Prediction wrapper-method 
        Requires X_test to be [n_samples,n,d], so use np.tile(X_test,[samples,1,1]) 
        before prediction.

        For policy search we use theano.scan. In that case we need to be able
        to feed in the input noise externally (provide_noise,noise)

        mode='symbolic' if we want to use this model as part of a larger graph(as in the policy search), 
        mode='numerical' for standard predictions, using compiled functions
        """
        print "HERE"
        X_test_n = (X_test - self.mean_X) / self.std_X
        
        #X_test_n = X_test

        if mode == 'symbolic':

            if provide_noise == True:
                # X_test_n.shape[0] refers to the number of samples, ie draws from the weight distribution
                m = self.bb_alpha.network.output_gn(X_test_n,noise,X_test_n.shape[0]) 
            else:
                m = self.bb_alpha.network.output(X_test_n,False,X_test_n.shape[0],use_indices=False) 

            log_v_noise = self.bb_alpha.network.log_v_noise
            noise_variance = T.tile(T.shape_padaxis(T.exp(log_v_noise[ 0, : ]), 0), [ m.shape[ 0 ], m.shape[ 1 ], 1])
            
        else:
            if X_test_n.ndim  == 2:
                X_test_n = np.tile(X_test_n,[self.params['samples'],1,1]) 
            
            m  = self.bb_alpha.fwpass(X_test_n,X_test_n.shape[0])
            log_v_noise = self.bb_alpha.network.log_v_noise.get_value()[0,:] 
            noise_variance = np.tile(np.exp(log_v_noise), [m.shape[0],m.shape[1],1])

        mt = m
        vt = noise_variance
        
        # TODO double check we don't need this?
        mt = mt * self.std_Y   + self.mean_Y
        vt *=  self.std_Y**2 
        return mt,vt

    def get_weights(self): 
        params = []

        # q(z|(x,y)) is only used during for training
        for p in self.bb_alpha.network.params:
            if p.name != 'log_v_z_par' and p.name != 'm_z_par':
                params.append(p.get_value())
        return params

