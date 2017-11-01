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
import numpy as np

import theano

import math

import theano.tensor as T

from network_layer import Network_layer

class Network:

    def __init__(self, layer_sizes, n_samples, v_prior, N):


        self.layer_sizes = layer_sizes
        self.N = N 
        self.v_prior = v_prior
        self.n_samples = n_samples
        self.log_v_noise = theano.shared(value =  -5 + np.zeros((1, layer_sizes[ -1 ])).astype(theano.config.floatX), name = 'log_v_noise', borrow = True)
        
        self.randomness_z = theano.shared(value = np.zeros((n_samples, N, 1)).astype(theano.config.floatX), name = 'z', borrow = True)
        self.v_prior_z = self.layer_sizes[ 0 ] - 1

        # We create the different layers

        self.layers = []
        for d_in, d_out, layer_type in zip(layer_sizes[ : -1 ], layer_sizes[ 1 : ], [ False ] * (len(layer_sizes) - 2) + [ True ]):
            self.layers.append(Network_layer(d_in, d_out, n_samples, v_prior, N, layer_type))

        self.mean_param_z = theano.shared(value = 0 * np.ones((self.N, 1)).astype(theano.config.floatX), name='m_z_par', borrow = True)
        self.log_var_param_z = theano.shared(value = 5 * np.ones((self.N, 1)).astype(theano.config.floatX), name='log_v_z_par', borrow = True)
        # We create the mean and variance parameters for all layers

        self.params = []
        for layer in self.layers:
            self.params.append(layer.mean_param_W)
            self.params.append(layer.log_var_param_W)
            self.params.append(layer.mean_param_b)
            self.params.append(layer.log_var_param_b)

        self.params.append(self.log_v_noise)
        self.params.append(self.mean_param_z)
        self.params.append(self.log_var_param_z)

        self.update_randomness()

    def log_f_hat_z(self):

        v_z = 1.0 / (1.0 / self.v_z - 1.0 / self.v_prior_z)
        m_z = self.m_z / self.v_z * v_z

        return (-0.5 * T.tile(1.0 / v_z, [ self.n_samples, 1, 1 ]) * self.z**2 + \
            T.tile(m_z / v_z, [ self.n_samples, 1, 1 ]) * self.z)[ :, :, 0 ]

    def log_normalizer_q_z(self):
        
        logZ_z = T.sum(0.5 * T.log(self.v_z * 2 * math.pi) + 0.5 * self.m_z**2 / self.v_z)
        return logZ_z


    def update_inputnoise(self,samples=0):
        if samples == 0:
            samples = self.n_samples
        self.randomness_z.set_value(np.float32(np.random.randn(samples, self.N, 1)))

    def update_weight_samples(self,samples=0):
        if samples == 0:
            samples = self.n_samples
        for layer in self.layers:
            layer.update_randomness(samples)

    def update_randomness(self,samples=0):
        self.update_inputnoise(samples)
        self.update_weight_samples(samples)

    def logistic(self, x):
        logi = 1.0 / (1.0 + T.exp(-x))
        return logi

    def output_gn(self,x,randomness_z,samples=0):
        if samples == 0:
            samples = self.n_samples
        #z = randomness_z[ : , 0 : x.shape[ 1 ], : ] *  T.tile(T.sqrt(self.v_prior_z), [samples,1,1])
        #x = T.concatenate((x,z), 2)
        x = T.concatenate((x,  randomness_z[ : , 0 : x.shape[ 1 ], : ]), 2)
        for layer in self.layers:
            x = layer.output(x,samples)
        return x


    def output(self, x,indexes,samples=0,use_indices=True):
        if samples == 0:
            samples = self.n_samples
        
        if use_indices == True:
            self.v_z = 1e-6 + self.logistic(self.log_var_param_z[ indexes, 0 : 1 ])*(self.v_prior_z - 2e-6)
            self.m_z = self.mean_param_z[ indexes, 0 : 1 ]
            self.z = self.randomness_z[ : , indexes, : ] * T.tile(T.sqrt(self.v_z), [ samples, 1, 1 ]) + T.tile(self.m_z, [ self.n_samples, 1, 1 ])
        else:
            self.z = self.randomness_z[:,0:x.shape[1],:] *  T.tile(T.sqrt(self.v_prior_z), [samples, 1, 1 ]) 

        x = T.concatenate((x,  self.z[ : , 0 : x.shape[ 1 ], : ]), 2)

        for layer in self.layers:
            x = layer.output(x,samples)

        return x

    def log_likelihood_values(self, x, y,indexes, location = 0.0, scale = 1.0):

        o = self.output(x,indexes)
        noise_variance = T.tile(T.shape_padaxis(T.exp(self.log_v_noise[ 0, : ]) * scale**2, 0), [ o.shape[ 0 ], o.shape[ 1 ], 1])
        location = T.tile(T.shape_padaxis(location, 0), [ o.shape[ 0 ], o.shape[ 1 ], 1])
        scale = T.tile(T.shape_padaxis(scale, 0), [ o.shape[ 0 ], o.shape[ 1 ], 1])
        return -0.5 * T.log(2 * math.pi * noise_variance) - \
            0.5 * (o * scale + location - T.tile(T.shape_padaxis(y, 0), [ o.shape[ 0 ], 1, 1 ]))**2 / noise_variance

    def log_normalizer_q(self):

        ret = 0
        for layer in self.layers:
            ret += layer.log_normalizer_q()

        return ret

    def log_Z_prior(self):

        n_weights = 0
        for layer in self.layers:
            n_weights += layer.get_n_weights()

        return n_weights * (0.5 * np.log(self.v_prior * 2 * math.pi))

    def log_f_hat(self):

        ret = 0
        for layer in self.layers:
            ret += layer.log_f_hat()

        return ret
