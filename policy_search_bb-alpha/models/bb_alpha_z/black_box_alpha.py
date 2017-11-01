# coding=utf-8
'''
The MIT License (MIT)

Copyright 2017 University of Cambridge, Siemens AG

Authors:  José Miguel Hernández-Lobato, Stefan Depeweg

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
import sys

import theano

import theano.tensor as T

import network

import numpy as np

import time

import copy


from timeit import default_timer as timer
from theano.compile.nanguardmode import NanGuardMode

def toString(x,digits=2):
    return str("".join(["{0:.",str(digits),"f}"]).format(np.round(x,digits)))
    #return np.round(x,digits)


def make_batches(N_data, batch_size):
    return [ slice(i, min(i + batch_size, N_data)) for i in range(0, N_data, batch_size) ]

def LogSumExp(x, axis = None):
    x_max = T.max(x, axis = axis, keepdims = True)
    return T.log(T.sum(T.exp(x - x_max), axis = axis, keepdims = True)) + x_max

def adam(loss, all_params, index_z, learning_rate = 0.001,rescale_local=10):

    # Standard Adam for all parameters except the last two (means and variancces of z)

    b1 = 0.9
    b2 = 0.999
    e = 1e-8
    gamma = 1 - 1e-8
    updates = []
    all_grads = theano.grad(loss, all_params)
    alpha = learning_rate

    for theta_previous, g in zip(all_params[ 0 : -2 ], all_grads[ 0 : -2 ]):
        m_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))
        v_previous = theano.shared(np.zeros(theta_previous.get_value().shape, dtype=theano.config.floatX))
        t = theano.shared(np.ones(theta_previous.get_value().shape, dtype=theano.config.floatX))
        m = b1 * m_previous + (1 - b1) * g                           # (Update biased first moment estimate)
        v = b2 * v_previous + (1 - b2) * g**2                            # (Update biased second raw moment estimate)
        m_hat = m / (1 - b1**t)                                          # (Compute bias-corrected first moment estimate)
        v_hat = v / (1 - b2**t)                                          # (Compute bias-corrected second raw moment estimate)
        theta = theta_previous - (alpha * m_hat) / (T.sqrt(v_hat) + e) #(Update parameters)
        updates.append((m_previous, m))
        updates.append((v_previous, v))
        updates.append((theta_previous, theta) )
        updates.append((t, t + 1.))

    # Specific Adam updates for the last two parameters (means and variancces of z)

    for theta_previous1, g1 in zip(all_params[ -2 : ], all_grads[ -2 : ]):
        #g1 = theano.gradient.grad_clip(g1,-1,1)

        #g = theano.printing.Print('g')(g)
        m_previous1 = theano.shared(np.zeros(theta_previous1.get_value().shape, dtype=theano.config.floatX))
        v_previous1 = theano.shared(np.zeros(theta_previous1.get_value().shape, dtype=theano.config.floatX))
        t1 = theano.shared(np.ones(theta_previous1.get_value().shape, dtype=theano.config.floatX))
        m1 = b1 * m_previous1 + (1 - b1) * g1                           # (Update biased first moment estimate)
        v1 = b2 * v_previous1 + (1 - b2) * g1**2                            # (Update biased second raw moment estimate)
        m_hat1 = m1 / (1 - b1**t1)                                          # (Compute bias-corrected first moment estimate)
        v_hat1 = v1 / (1 - b2**t1)                                          # (Compute bias-corrected second raw moment estimate)
        theta1 = theta_previous1 - (alpha * rescale_local * m_hat1) / (T.sqrt(v_hat1) + e) #(Update parameters)


        updates.append((m_previous1, T.set_subtensor(m_previous1[ index_z ], m1[ index_z ])))
        updates.append((v_previous1, T.set_subtensor(v_previous1[ index_z ], v1[ index_z ])))
        updates.append((theta_previous1, T.set_subtensor(theta_previous1[ index_z ], theta1[ index_z ])))
        updates.append((t1, T.set_subtensor(t1[ index_z ], t1[ index_z ] + 1.)))

    return updates

class BB_alpha:

    def __init__(self, layer_sizes, n_samples, alpha, learning_rate, v_prior, batch_size, X_train, y_train, N_train,
                xtest=None, ytest=None, ntest=None):
        

        layer_sizes = copy.copy(layer_sizes)
        layer_sizes[ 0 ] = layer_sizes[ 0 ] + 1
        print layer_sizes
        self.batch_size = batch_size
        self.N_train = N_train
        self.X_train = X_train
        self.y_train = y_train
        
        self.X_test = xtest
        self.Y_test = ytest
        self.N_test = ntest

        self.rate = learning_rate

        # We create the network

        self.network = network.Network(layer_sizes, n_samples, v_prior, N_train)

        # index to a batch

        index = T.lscalar()
        self.indexes = T.vector('index', dtype = 'int32')
        #self.indexes_test = T.vector('index_test', dtype = 'int32')
        indexes_train = theano.shared(value = np.array(range(0, N_train), dtype = np.int32), borrow = True)
        indexes_test = theano.shared(value = np.array(range(0, ntest), dtype = np.int32), borrow = True)

        self.x = T.tensor3('x',dtype=theano.config.floatX)
        self.y = T.matrix('y', dtype =theano.config.floatX)
        self.lr = T.fscalar()

        # The logarithm of the values for the likelihood factors
        sampl = T.bscalar()
        self.fwpass = theano.function(outputs=self.network.output(self.x,False,samples=sampl,use_indices=False), inputs=[self.x,sampl],allow_input_downcast=True)
        
        ll_train = self.network.log_likelihood_values(self.x, self.y, self.indexes, 0.0, 1.0)


        self.estimate_marginal_ll = (-1.0 * N_train / (self.x.shape[ 1 ] * alpha) * \
            T.sum(LogSumExp(alpha * (T.sum(ll_train, 2) - self.network.log_f_hat() - self.network.log_f_hat_z()), 0)+ \
                T.log(1.0 / n_samples)) - self.network.log_normalizer_q() - 1.0 * N_train / self.x.shape[ 1 ] * self.network.log_normalizer_q_z() + \
            self.network.log_Z_prior())
        
        # We create a theano function for updating q
        upd = adam(self.estimate_marginal_ll, self.network.params,indexes_train[index*batch_size:(index+1)*batch_size],self.rate,rescale_local=np.float32(N_train/batch_size))

        self.process_minibatch = theano.function([ index], self.estimate_marginal_ll, \
            updates = upd, \
            givens = { self.x: T.tile(self.X_train[ index * batch_size: (index + 1) * batch_size] , [ n_samples, 1, 1 ]),
            self.y: self.y_train[ index * batch_size: (index + 1) * batch_size ],
            self.indexes: indexes_train[ index * batch_size : (index + 1) * batch_size ] })

        # We create a theano function for making predictions


        self.error_minibatch_train = theano.function([ index ],
            T.sum((T.mean(self.network.output(self.x,self.indexes), 0, keepdims = True)[ 0, :, : ] - self.y)**2) / layer_sizes[ -1 ],
            givens = { self.x: T.tile(self.X_train[ index * batch_size: (index + 1) * batch_size ], [ n_samples, 1, 1 ]),
            self.y: self.y_train[ index * batch_size: (index + 1) * batch_size ],
            self.indexes: indexes_train[ index * batch_size : (index + 1) * batch_size ] })

        self.ll_minibatch_train = theano.function([ index ], T.sum(LogSumExp(T.sum(ll_train, 2), 0) + T.log(1.0 / n_samples)), \

            givens = { self.x: T.tile(self.X_train[ index * batch_size: (index + 1) * batch_size ], [ n_samples, 1, 1 ]),
            self.y: self.y_train[ index * batch_size: (index + 1) * batch_size ],
            self.indexes: indexes_train[ index * batch_size : (index + 1) * batch_size ] })
        
       
    
    # TODO create new functions for self.error_minibatch_test, self.ll_minibatch_test -- same as for train, just switch 
        # the dataset that's being used
        
        #if self.X_test is not None and self.Y_test is not None:
        self.error_minibatch_test = theano.function([ index ],
                T.sum((T.mean(self.network.output(self.x,self.indexes), 0, keepdims = True)[ 0, :, : ] - self.y)**2) 
                                                                                                      / layer_sizes[ -1 ],
                givens = { self.x: T.tile(self.X_test[index * batch_size: (index + 1) * batch_size ], [ n_samples, 1, 1 ]),
                self.y: self.Y_test[ index * batch_size: (index + 1) * batch_size ],
                self.indexes: indexes_test[ index * batch_size : (index + 1) * batch_size ] }, on_unused_input='warn')

        self.ll_minibatch_test = theano.function([ index ], T.sum(LogSumExp(T.sum(ll_train, 2), 0) + T.log(1.0 / n_samples)),
            givens = { self.x: T.tile(self.X_test[ index * batch_size: (index + 1) * batch_size ], [ n_samples, 1, 1 ]),
                self.y: self.Y_test[ index * batch_size: (index + 1) * batch_size ],
                self.indexes: indexes_test[ index * batch_size : (index + 1) * batch_size ] }, on_unused_input='warn')



    def train(self, n_epochs):
        n_batches_train = np.int(np.ceil(1.0 * self.N_train / self.batch_size))
        n_batches_test = np.int(np.ceil(1.0 * self.N_test / self.batch_size))
        i = 0 

        while i < n_epochs:
            start = timer()
            energy = []
            permutation = np.random.choice(range(n_batches_train), n_batches_train, replace = False)
            for idxs in range(n_batches_train):
                if idxs % 10 == 9:
                    self.network.update_randomness()
                ret = self.process_minibatch(permutation[ idxs ])
                energy.append(ret)

            # We evaluate the performance on the test data
            # TODO copy the block of code below and the giant print statement, and call functions for the test set
            # Do this every M epochs (not on completion of 1)

            self.network.update_randomness()
            error_train = 0
            ll_train = 0
            for idxs in range(n_batches_train):
                error_train += self.error_minibatch_train(idxs)
                ll_train += self.ll_minibatch_train(idxs)
            error_train /= self.N_train
            #error_train = np.sqrt(error_train)
            ll_train /= self.N_train
            end = timer()
            i += 1

            print ', '.join([str(i)+'/'+str(n_epochs),toString(error_train,3),toString(ll_train,2),toString(np.mean(energy),2),toString(end-start,1),toString(np.mean(self.network.mean_param_z.get_value()),2),toString(np.mean(self.network.log_var_param_z.get_value()),2),toString(self.network.log_v_noise.get_value()[0,0])])
            
            if i > 0 and i % 50 == 0 and self.X_test is not None and self.Y_test is not None:
                error_test = 0
                ll_test = 0
                for idxs in range(n_batches_test):
                    error_test += self.error_minibatch_test(idxs)
                    ll_test += self.ll_minibatch_test(idxs)
                error_test /= self.N_test
                ll_test /= self.N_test
                
                print ', '.join([toString(error_test,3),toString(ll_test,2)])
                
                    
                    
