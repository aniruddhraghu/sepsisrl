"""
A simple version of Proximal Policy Optimization (PPO) using single thread.

Based on:
1. Emergence of Locomotion Behaviours in Rich Environments (Google Deepmind): [https://arxiv.org/abs/1707.02286]
2. Proximal Policy Optimization Algorithms (OpenAI): [https://arxiv.org/abs/1707.06347]

View more on my tutorial website: https://morvanzhou.github.io/tutorials
"""

import tensorflow as tf
import numpy as np
import os

from env_model import EnvModel

N_EPOCHS = 1000
EP_LEN = 10
GAMMA = 0.99
A_LR = 0.0001
C_LR = 0.0002
BATCH = 32
A_UPDATE_STEPS = 10
C_UPDATE_STEPS = 10
S_DIM, A_DIM = 198, 1
METHOD = [
    dict(name='kl_pen', kl_target=0.01, lam=0.5),   # KL penalty
    dict(name='clip', epsilon=0.2),                 # Clipped surrogate objective, find this is better
][1]        # choose the method for optimization

class PPO(object):

    def __init__(self,sess):
        self.sess = sess
        self.tfs = tf.placeholder(tf.float32, [None, S_DIM], 'state')
        self.phase = tf.placeholder(tf.bool)

        # critic
        with tf.variable_scope('critic'):
            l1 = tf.layers.dense(self.tfs, 100, tf.nn.relu)
            self.v = tf.layers.dense(l1, 1)
            self.tfdc_r = tf.placeholder(tf.float32, [None, 1], 'discounted_r')
            self.advantage = self.tfdc_r - self.v
            self.closs = tf.reduce_mean(tf.square(self.advantage))
            self.ctrain_op = tf.train.AdamOptimizer(C_LR).minimize(self.closs)

        # actor
        pi, pi_params = self._build_anet('pi', trainable=True)
        oldpi, oldpi_params = self._build_anet('oldpi', trainable=False)
        with tf.variable_scope('sample_action'):
            self.sample_op = tf.squeeze(pi.sample(1), axis=0)       # choosing action
        with tf.variable_scope('update_oldpi'):
            self.update_oldpi_op = [oldp.assign(p) for p, oldp in zip(pi_params, oldpi_params)]

        self.tfa = tf.placeholder(tf.float32, [None, A_DIM], 'action')
        self.tfadv = tf.placeholder(tf.float32, [None, 1], 'advantage')
        with tf.variable_scope('loss'):
            with tf.variable_scope('surrogate'):
                # ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                ratio = tf.exp(pi.log_prob(self.tfa) - oldpi.log_prob(self.tfa))
                surr = ratio * self.tfadv
            if METHOD['name'] == 'kl_pen':
                self.tflam = tf.placeholder(tf.float32, None, 'lambda')
                kl = tf.distributions.kl_divergence(oldpi, pi)
                self.kl_mean = tf.reduce_mean(kl)
                self.aloss = -(tf.reduce_mean(surr - self.tflam * kl))
            else:   # clipping method, find this is better
                self.aloss = -tf.reduce_mean(tf.minimum(
                    surr,
                    tf.clip_by_value(ratio, 1.-METHOD['epsilon'], 1.+METHOD['epsilon'])*self.tfadv))

        with tf.variable_scope('atrain'):
            self.atrain_op = tf.train.AdamOptimizer(A_LR).minimize(self.aloss)

        tf.summary.FileWriter("log/", self.sess.graph)

        self._initialize_uninitialized()

        # INITIALISE THE POLICY NETWORK WITH THE BEHAVIOUR CLONING WEIGHTS

        self._initialise_with_behaviour_clone()

    def update(self, s, a, r):
        self.sess.run(self.update_oldpi_op)
        adv = self.sess.run(self.advantage, {self.tfs: s, self.tfdc_r: r})
        # adv = (adv - adv.mean())/(adv.std()+1e-6)     # sometimes helpful

        # update actor
        if METHOD['name'] == 'kl_pen':
            for _ in range(A_UPDATE_STEPS):
                _, kl = self.sess.run(
                    [self.atrain_op, self.kl_mean],
                    {self.tfs: s, self.tfa: a, self.tfadv: adv, self.tflam: METHOD['lam']})
                if kl > 4*METHOD['kl_target']:  # this in in google's paper
                    break
            if kl < METHOD['kl_target'] / 1.5:  # adaptive lambda, this is in OpenAI's paper
                METHOD['lam'] /= 2
            elif kl > METHOD['kl_target'] * 1.5:
                METHOD['lam'] *= 2
            METHOD['lam'] = np.clip(METHOD['lam'], 1e-4, 10)    # some time explode, this is my method
        else:   # clipping method, find this is better (OpenAI's paper)
            [self.sess.run(self.atrain_op, {self.tfs: s, self.tfa: a, self.tfadv: adv}) for _ in range(A_UPDATE_STEPS)]

        # update critic
        [self.sess.run(self.ctrain_op, {self.tfs: s, self.tfdc_r: r}) for _ in range(C_UPDATE_STEPS)]

    def _build_anet(self, name, trainable):
        with tf.variable_scope(name):
            fc_1 = tf.contrib.layers.fully_connected(self.tfs, 64, activation_fn=tf.nn.relu, trainable=trainable)
            fc_2 = tf.contrib.layers.fully_connected(fc_1 , 64, activation_fn=tf.nn.relu, trainable=trainable)
            logits = tf.contrib.layers.fully_connected(fc_2 , 25, activation_fn=None, trainable=trainable)
            probs = tf.nn.softmax(logits)
            output = tf.contrib.distributions.Categorical(probs=probs)
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=name)
        return output, params

    def choose_action(self, s):
        if s.ndim< 2: s = s[np.newaxis, :]
        a = self.sess.run(self.sample_op, {self.tfs: s})
        return np.clip(a, -2, 2)

    def get_v(self, s):
        if s.ndim < 2: s = s[np.newaxis, :]
        return self.sess.run(self.v, {self.tfs: s})


    def _initialize_uninitialized(self):
        global_vars = tf.global_variables()
        is_not_initialized = self.sess.run([tf.is_variable_initialized(var) for var in global_vars])
        not_initialized_vars = [v for (v, f) in zip(global_vars, is_not_initialized) if not f]

        print ([str(i.name) for i in not_initialized_vars]) # only for testing
        if len(not_initialized_vars):
            self.sess.run(tf.variables_initializer(not_initialized_vars))

    def _initialise_with_behaviour_clone(self):
        params = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope='pi')

        saved_names = [u'fully_connected/weights', u'fully_connected/biases', 
                    u'fully_connected_1/weights', u'fully_connected_1/biases', u'fully_connected_2/weights', u'fully_connected_2/biases']

        restore_dict = {i:v for (i,v) in zip(saved_names, params)}
        restorer = tf.train.Saver(restore_dict)
        # TODO sort out the path below
        restorer.restore(self.sess, 'behaviour_clone/ckpt')



def load_data():
    dire = 'converted_data/'

    train_feat_zeros = np.loadtxt(dire + 'X_train_hist_zeros.txt')
    print ("Loaded train_zeros")

    val_feat_zeros = np.loadtxt(dire + 'X_val_hist_zeros.txt')
    print ("Loaded val_zeros")

    return train_feat_zeros, val_feat_zeros


def main():
    batch_size = 512
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True  # Don't use all GPUs 
    config.allow_soft_placement = True  # Enable manual control
    X_train, X_val = load_data()
    with tf.Session(config=config) as sess:
        env = EnvModel(sess)
        ppo = PPO(sess)
        all_ep_r = []

        for ep in range(N_EPOCHS):
            X_train_shuffle = np.random.permutation(X_train)
            X_train_shuffle = X_train_shuffle[:, :-2] # remove the last two dimensions which represent the actions taken in the dataset
            assert X_train_shuffle.shape[1] == 198
            buffer_s, buffer_a, buffer_r = [], [], []
            ep_r = 0
            start_idx = 0
            end_idx = 0
            while start_idx < len(X_train_shuffle):
                end_idx = min(len(X_train_shuffle),start_idx + batch_size)
                s = X_train_shuffle[start_idx:end_idx]
                for t in range(EP_LEN):    # in one episode
                    # print("SHAPE ", s.shape)
                    a = ppo.choose_action(s)
                    # print("ACTION SHAPE ", a.shape)
                    # a = ppo.choose_action(s)
                    if t == EP_LEN-1:
                        terminal = True
                    else:
                        terminal = False
                    s_, r  = env.step(s,a,terminal)
                    buffer_s.append(s)
                    buffer_a.append(a)
                    buffer_r.append(r)    # normalize reward, find to be useful
                    s = s_
                    ep_r += np.mean(r)

                    # update ppo
                    if (t+1) % BATCH == 0 or t == EP_LEN-1:
                        v_s_ = ppo.get_v(s_)
                        # print( v_s_.shape)
                        discounted_r = []
                        for r in buffer_r[::-1]:
                            v_s_ = r + GAMMA * v_s_
                            # print(r.shape, v_s_.shape)
                            discounted_r.append(v_s_)
                        discounted_r.reverse()
                        # print(np.array(buffer_s).shape,np.array(buffer_a).shape, np.array(discounted_r).shape )
                        bs = np.reshape(np.array(buffer_s), [-1, 198])
                        ba = np.reshape(np.array(buffer_a),[-1,1] )
                        br = np.reshape(np.array(discounted_r), [-1,1])
                        buffer_s, buffer_a, buffer_r = [], [], []
                        ppo.update(bs, ba, br)
                start_idx += batch_size
                print(start_idx)
            if ep == 0: all_ep_r.append(ep_r)
            else: all_ep_r.append(all_ep_r[-1]*0.9 + ep_r*0.1)
            print(
                'Ep: %i' % ep,
                "|Ep_r: %i" % ep_r,
                ("|Lam: %.4f" % METHOD['lam']) if METHOD['name'] == 'kl_pen' else '',
            )

    # plt.plot(np.arange(len(all_ep_r)), all_ep_r)
    # plt.xlabel('Episode');plt.ylabel('Moving averaged episode reward');plt.show()

if __name__ == '__main__':
    main()