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

import lasagne
from lasagne.updates import total_norm_constraint
import lasagne.utils as utils

class PolicySearch:
    def __init__(self,params,params_task,X,model,policy):
        # Note: X passed to the constructor is of dimension (DATASETSIZE, (state_vec_size + num_actions). 
        # This is required by the BNN when it predicts the next state and mortality
        # The PolicyNetwork relies on X[:, :-2] (everything except the action taken).
        
        self.rng = np.random.RandomState()

        self.model = model
        self.policy = policy

        self.params = params
        self.params_task = params_task

        # like a placeholder in tf?
        self.x = T.matrix('x')
        cost  =  self.control(self.x)

        self.fwpass  = theano.function(inputs=[self.x], outputs = cost,allow_input_downcast=True)
        self.train_func = theano.function(inputs=[self.x],outputs=[cost], updates=self.adam(cost,lasagne.layers.get_all_params(self.policy,trainable=True),learning_rate=self.params['learning_rate']), allow_input_downcast=True)

        self.policy_network = theano.function(inputs=[self.x],outputs=self.predict(self.x))


    def control(self,st):
        # st is of dimension batchsize * (state_vec_size + num_actions)
        # Need to cut off the last two dimensions for the policy network prediction
        srng = T.shared_randomstreams.RandomStreams(self.rng.randint(999999))
        # do n roll-outs for each starting state
        n = self.params['samples']
        
        # st_s is of dimension (batchsize*n), (state_vec_size+num_actions)
        st_s = T.tile(st,[n,1])
        
        # onoise should be of size st_s.shape[0] * (st_s.shape[1]-1) * timesteps
        # st_s.shape[1] contains the actions too; we don't want these included, but we do want the mortality flag
        onoise =  srng.normal(size=(st_s.shape[0],st_s.shape[1]-1,self.params['T']))
        
        # inoise is of shape: n_samples * batch_size * timesteps. Not sure about this!
        inoise = T.sqrt(st.shape[1]) * srng.normal(size=(n,st.shape[0],self.params['T']))
        
         
        # TODO check the dimensionality of this: is it correct?
        rew_init = np.zeros(self.params['batchsize']*n)
        ([_, rewards, _, term_rewards], updates) = theano.scan(fn=self._step, 
                                                            outputs_info = [st_s,T.as_tensor_variable(rew_init),
                                                             T.as_tensor_variable(0), None],
                                                            non_sequences=[onoise,inoise],
                                                            n_steps = self.params['T'])
        
        # Some extra code to compute the actual reward we want: need to check that the timesteps and dimensionalites are ok.
        # -1 corresponds to no mortality; +1 corresponds to mortality, so multiply by -15
        summed_reward = rewards[-1,:]
        term_rewards = term_rewards[-1,:] * (-15)
        
        final_cost = -1*(summed_reward + term_rewards)
        
        return final_cost.mean()

   
    def _step(self,st_s,rew,t,onoise,inoise):
        # st_s is of dimension (batchsize*n), (state_vec_size+num_actions)
        # st_s[:, :-2] will extract everything but the actions out

            
        on_t =  onoise[:,:,t]
        in_t = inoise[:,:,t:t+1]
        
        # get action
        #CHECK THIS DIMENSION
        pol_st_s = st_s[:,:-2]
        
        at_s = self.predict(pol_st_s)
               
        # at_s is of dimension (n*batchsize, 2). Need to create state_new = concat_1_axis(st_s[:,:-2],at_s)
        bnn_st_at = T.concatenate([pol_st_s, at_s], axis=1)
        
        #Get delta_t from BNN. Note that the BNN output dimension is (statevec_size + 1) 
        
        #reshape into num_samples * batchsize * (statevec_size+2). 
        #First dimension of corresponds to number of BNN samples drawn
        x_bnn = bnn_st_at.reshape((self.params['samples'],bnn_st_at.shape[0]/self.params['samples'],bnn_st_at.shape[1]))
        
        # obtain mean and variance
        delta_x_bnn_mean, delta_x_bnn_var = self.model.predict(x_bnn, mode='symbolic',provide_noise=True,noise=in_t)
        
        #reshape back
        delta_x_bnn_mean= delta_x_bnn_mean.reshape((delta_x_bnn_mean.shape[0]*delta_x_bnn_mean.shape[1],
                                                    delta_x_bnn_mean.shape[2]))
        delta_x_bnn_var = delta_x_bnn_var.reshape((delta_x_bnn_var.shape[0]*delta_x_bnn_var.shape[1],
                                                   delta_x_bnn_var.shape[2]))

       
        # Sample from output noise
        delta_x_bnn = on_t * T.sqrt(delta_x_bnn_var) + delta_x_bnn_mean
        
        #delta_x_bnn is of shape (n*batchsize, state_vec+1). Extract out the mortality flag and the actual state change
        state_change = delta_x_bnn[:,:-1]
        mort_flag = delta_x_bnn[:,-1]
        
        tmp = pol_st_s + state_change
        
        # obtain the new state
        new_st_s = T.concatenate([tmp, at_s], axis=1)
        
        
        # compute the reward:
        # some constants. These are changed from earlier!
        c0 = -1.0/4
        c1 = -0.5/4
        c2 = -2
        
        #grab the right cols
        sofa_now = st_s[:, self.params['sofa_index']]
        lact_now = st_s[:, self.params['lact_index']]
        
        sofa_next = new_st_s[:, self.params['sofa_index']]
        lact_next = new_st_s[:, self.params['lact_index']]
        
        # unscale the sofa/lactate
        sofa_now = sofa_now * self.params['sofa_std'] + self.params['sofa_mean']
        sofa_next = sofa_next * self.params['sofa_std'] + self.params['sofa_mean']
        
        lact_now = lact_now * self.params['lact_std'] + self.params['lact_mean']
        lact_next = lact_next * self.params['lact_std'] + self.params['lact_mean']
        
        #reward function: R(t) = c0*(T.tanh(sofa_now)) + c1(sofa_next-sofa_now) + c2*T.tanh(lact_next-lact_now)
        # TODO do we want to tanh the change in SOFA score too?
        rew_t = c0*T.tanh(sofa_now) + c1*(sofa_next-sofa_now) + c2*T.tanh(lact_next-lact_now)

        return [new_st_s.astype('float32'),rew+rew_t,t+1, mort_flag]


    def predict(self,X):
        # Commented out normalisation as this has already been completed.
        #X = (X - self.model.mean_X.astype(theano.config.floatX)) / self.model.std_X.astype(theano.config.floatX)
        return lasagne.layers.get_output(self.policy,X)
    

    def adam(self,cost, params, learning_rate=0.001, beta1=0.9,
             beta2=0.999, epsilon=1e-8):

        all_grads = T.grad(cost=cost, wrt=params)
        all_grads = total_norm_constraint(all_grads,10)
    
        grad_norm = T.sqrt(sum(map(lambda x: T.sqr(x).sum(), all_grads)))
        not_finite = T.or_(T.isnan(grad_norm), T.isinf(grad_norm))
    
        t_prev = theano.shared(utils.floatX(0.))
        updates = OrderedDict()
    
        t = t_prev + 1
        a_t = learning_rate*T.sqrt(1-beta2**t)/(1-beta1**t)
    
        for param, g_t in zip(params, all_grads):
            g_t = T.switch(not_finite, 0.1 * param,g_t)
            value = param.get_value(borrow=True)
            m_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
            v_prev = theano.shared(np.zeros(value.shape, dtype=value.dtype),
                                   broadcastable=param.broadcastable)
    
            m_t = beta1*m_prev + (1-beta1)*g_t
            v_t = beta2*v_prev + (1-beta2)*g_t**2
            step = a_t*m_t/(T.sqrt(v_t) + epsilon)
    
            updates[m_prev] = m_t
            updates[v_prev] = v_t
            updates[param] = param - step
    
        updates[t_prev] = t
        return updates
    
