"""
Inverse reinforcement learning with policy improvement in the loop
Uses the MaxEnt IRL guided cost learning algorithm in Finn et al. 2016
"""

import numpy as np
from numpy.linalg import norm
import tensorflow as tf

from scipy import special
from scipy.stats import entropy
from scipy.stats import gaussian_kde
# from scipy.stats import dirichlet
import functools

import platform
if (platform.system() == "Windows"):
    import pandas as pd
    import matplotlib.pylab as plt
    from matplotlib.backends.backend_pdf import PdfPages
    from pylab import flipud
    import var

import os
import itertools
import time
import random

import networks

class AC_IRL:

    def __init__(self, theta=8.86349, shift=0.16, alpha_scale=12000, d=15, lr_reward=1e-4, num_policies=10, c=2e11, reg='dropout_l1l2', n_fc3=8, n_fc4=4, saved_network=None, use_tf=True, summarize=False):
        """
        reg - 'none', 'dropout', 'l1l2', 'dropout_l1l2'
        use_tf - if True, create tensorflow graphs as usual, else do not instantiate graph
        """
        self.summarize = summarize
        if (platform.system() == "Windows"):
            self.var = var.var(d=d)

        # initialize theta
        self.theta = theta
        # remember a initial theta as a reset value to use during IRL
        self.theta_initial = theta
        self.shift = shift
        self.alpha_scale = alpha_scale
        # initialize weight vector (column) for value function approximation
        self.w = self.init_w(d)
        # number of topics
        self.d = d

        # learning rate for reward optimizer
        self.lr_reward = lr_reward
        # number of policies to record
        self.num_policies = num_policies
        # normalizer
        self.c = c
        # regularization
        self.reg = reg
        self.n_fc3 = n_fc3
        self.n_fc4 = n_fc4

        # initialize collection of start states
        self.init_pi0(path_to_dir=os.getcwd()+'/train_normalized_round2')
        self.num_start_samples = self.mat_pi0.shape[0] # number of rows

        # Will become list of list of tuples of the form (state, action)
        self.list_demonstrations = self.read_demonstrations(state_dir='./train_normalized_round2', action_dir='./actions_2', dim_action=20, start_day=1)
        self.list_demonstrations_test = self.read_demonstrations(state_dir='./test_normalized_round2', action_dir='./actions_test_2', dim_action=20, start_day=19)
        # Collect a set of transitions from demo trajectories for testing reward function
        self.list_eval_demo_transitions = [pair for traj in self.list_demonstrations for pair in traj]
        # self.list_eval_demo_transitions = self.get_eval_transitions(self.list_demonstrations)

        # This is D_samp in the IRL algorithm. Will be populated while running outerloop()
        self.list_generated = []

        # Create neural net representation of reward function
        if use_tf:
            self.create_network()

        # number of demonstration trajectories to sample each time for reward learning
        self.num_demo_samples = 5
        # number of generated trajectories to sample each time for reward learning
        self.num_gen_samples = 5
        # number of trajectories to use for part 2 of loss function
        # self.num_sampled_trajectories = self.num_demo_samples + self.num_gen_samples
        self.num_sampled_trajectories = self.num_gen_samples
        # list of policies parameterized by theta, beginning with the initialized theta
        self.list_policies = [theta] * self.num_policies

        if use_tf:
            # Define gradient descent steps for reward learning
            self.create_training_method()
            
            self.sess = tf.Session()
            
            if summarize:
                self.merged = tf.summary.merge_all()
                self.train_writer = tf.summary.FileWriter('./log', self.sess.graph)
            
            init = tf.global_variables_initializer()
            self.sess.run(init)
            
            self.saver = tf.train.Saver()
            
            if saved_network:
                self.saver.restore(self.sess, "saved/"+saved_network)
            
        
    # ------------------- File processing functions ------------------ #

    def convert_action(self, state_dir, action_dir, action_write_dir, dim_action=20, start_day=1):
        """
        For each action matrix, if row is all zeros except single 1 on the diagonal entry, 
        check whether the person actually stayed in topic, or whether 
        the number of people was zero and the 1 was added artificially during recording

        dim_action - dimension of action matrix that was recorded (will be larger than or equal to self.d)
        """
        num_file_action = len(os.listdir(action_dir))
        num_file_state = len(os.listdir(state_dir))
        if num_file_action != num_file_state:
            print("Weird")
        list_demonstrations = []
        for idx_day in range(start_day, start_day+num_file_action):        
            file_state = "trend_distribution_day%d.csv" % idx_day
            file_action = "action_day%d.txt" % idx_day
            # read states for this day
            states = pd.read_table(state_dir+'/'+file_state, delimiter=' ', header=None)
            # convert to np matrix
            states = states.as_matrix() # 16 x d
            # read actions for this day, automatically skips over blank lines
            actions = pd.read_table(action_dir+'/'+file_action, delimiter=' ', header=None)
            actions = actions.as_matrix() # (15*d) * d

            f = open(action_write_dir+'/action_day%d.txt' % idx_day, 'a')
            for hour in range(0,15):
                state = states[hour, 0:self.d]
                # state_next = states[hour+1, 0:self.d]
                # read everything
                action = actions[hour*dim_action:(hour+1)*dim_action, :]
                for topic in range(0, self.d):
                    if action[topic, topic] == 1.0 and state[topic] == 0:
                        # keep same number of entries, but normalize against self.d
                        action[topic, :] = np.ones(dim_action) / self.d
                    action[topic].tofile(f, sep=' ', format='%.3e')
                    f.write('\n')
                # write the rest without change
                for topic in range(self.d, dim_action):
                    action[topic].tofile(f, sep=' ', format='%.3e')
                    f.write('\n')
                f.write('\n')
            f.close()

    # ------------------- End file processing functions ------------------ #


    # ------------------- New and overridden functions ---------------- #

    def read_demonstrations(self, state_dir, action_dir, dim_action=20, start_day=1):
        """
        Reads measured trajectories to produce list of trajectories,
        where each trajectory is a list of (state, action) pairs,
        where each state is a vector and action is a transition matrix
        
        state_dir - name of folder containing measured states for each hour of each day
        action_dir - name of folder containing measured actions for each hour of each day
        dim_action - dimension of action matrix that was recorded (will be larger than or equal to self.d)
        start_day - day number, 1 for train group, some other number for test group
        """
        print("Inside read_demonstrations")
        num_file_action = len(os.listdir(action_dir))
        num_file_state = len(os.listdir(state_dir))
        if num_file_action != num_file_state:
            print("Weird")
        list_demonstrations = []
        for idx_day in range(start_day, start_day+num_file_action):
            trajectory = []
            file_state = "trend_distribution_day%d.csv" % idx_day
            file_action = "action_day%d.txt" % idx_day
            # read states for this day
            states = pd.read_table(state_dir+'/'+file_state, delimiter=' ', header=None)
            # convert to np matrix
            states = states.as_matrix() # 16 x d
            # read actions for this day, automatically skips over blank lines
            actions = pd.read_table(action_dir+'/'+file_action, delimiter=' ', header=None)
            actions = actions.as_matrix() # (15*d) * d
            # group into (state, action) pairs
            for hour in range(0,15):
                state = states[hour, 0:self.d]
                action = actions[hour*dim_action:(hour*dim_action+self.d), 0:self.d]
                # append state-action pair
                trajectory.append( (state, action) )
            # append to list of demonstrations
            list_demonstrations.append( trajectory )
        return list_demonstrations


    def get_eval_transitions(self, list_trajectories):
        """
        Returns a list of (s,a) tuples, one tuple from each input trajectory in 
        the input list

        list_trajectories - list of list of (s,a) tuples
        """
        list_test_transitions = []
        num_demos = len(list_trajectories)
        for idx in range(num_demos):
            traj = list_trajectories[idx]
            # Select the transition at index equal to the trajectory index, mod traj length
            # This is just a deterministic selection, out of many possible selection schemes
            idx_transition = idx % 15
            list_test_transitions.append( traj[idx_transition] )

        return list_test_transitions


    def get_trainable_var_under(self, scope_name):
        """
        Returns list of trainable variables nested under scope_name
        """
        list_vars = tf.trainable_variables()
        filtered_list = [var for var in list_vars if scope_name in var.name]

        return filtered_list


    def create_network(self):
        """
        Creates neural net representation of reward function
        """
        print("Inside create_network")
        # placeholder for actions in demonstration batch, shape [N, num_actions, d,d]
        # where N = number of trajectories * num actions along trajectory (should be 15)
        self.demo_actions = tf.placeholder(dtype=tf.float32, shape=[None,self.d,self.d], name='demo_actions')
        # placeholder for states in demonstration batch
        # where N = number of trajectories * num states along trajectory (should be 15)
        self.demo_states = tf.placeholder(dtype=tf.float32, shape=[None,self.d], name='demo_states')
        # placeholder for actions in generated batch
        self.gen_actions = tf.placeholder(dtype=tf.float32, shape=[None,self.d,self.d], name='gen_actions')
        # placeholder for states in generated batch
        self.gen_states = tf.placeholder(dtype=tf.float32, shape=[None,self.d], name='gen_states')
        with tf.variable_scope("reward") as scope:
            if self.reg == 'none':
                # rewards for state-action pairs in demonstration batch
                # expected dimension [N, 1] where N = total number of transitions in batch
                self.reward_demo = networks.r_net(self.demo_states, self.demo_actions, f1=1, k1=5, f2=2, k2=3, n_fc3=self.n_fc3, n_fc4=self.n_fc4, d=self.d)
                scope.reuse_variables()
                # rewards for state-action pairs in generated batch
                # expected dimension [N, 1] where N = total number of transitions in batch
                self.reward_gen = networks.r_net(self.gen_states, self.gen_actions, f1=1, k1=5, f2=2, k2=3, n_fc3=self.n_fc3, n_fc4=self.n_fc4, d=self.d)
            elif self.reg == 'dropout':
                self.reward_demo = networks.r_net_dropout(self.demo_states, self.demo_actions, f1=1, k1=5, f2=2, k2=3, n_fc3=self.n_fc3, n_fc4=self.n_fc4, d=self.d)
                scope.reuse_variables()
                self.reward_gen = networks.r_net_dropout(self.gen_states, self.gen_actions, f1=1, k1=5, f2=2, k2=3, n_fc3=self.n_fc3, n_fc4=self.n_fc4, d=self.d)
            elif self.reg == 'l1l2':
                self.reward_demo = networks.r_net_l1l2(self.demo_states, self.demo_actions, f1=1, k1=5, f2=2, k2=3, n_fc3=self.n_fc3, n_fc4=self.n_fc4, d=self.d)
                scope.reuse_variables()
                self.reward_gen = networks.r_net_l1l2(self.gen_states, self.gen_actions, f1=1, k1=5, f2=2, k2=3, n_fc3=self.n_fc3, n_fc4=self.n_fc4, d=self.d)
            elif self.reg == 'dropout_l1l2':
                self.reward_demo = networks.r_net_dropout_l1l2(self.demo_states, self.demo_actions, f1=1, k1=5, f2=2, k2=3, n_fc3=self.n_fc3, n_fc4=self.n_fc4, d=self.d)
                scope.reuse_variables()
                self.reward_gen = networks.r_net_dropout_l1l2(self.gen_states, self.gen_actions, f1=1, k1=5, f2=2, k2=3, n_fc3=self.n_fc3, n_fc4=self.n_fc4, d=self.d)
                

    def calc_pdf_action(self, theta, action, state):
        """
        Evaluates q(a_t; s_t, theta) where q is Dirichlet policy parameterized by theta
        theta - real value
        action - TF matrix P_{ij}
        state - TF vector pi_i
        """
        # compute alpha^i_j
        # mat1 = np.repeat(state.reshape(1, self.d), self.d, 0) # all rows same
        mat1 = tf.tile( tf.reshape(state, [1,self.d]), [self.d, 1] ) # all rows same
        # mat2 = np.repeat(state.reshape(self.d, 1), self.d, 1) # all columns same
        mat2 = tf.tile( tf.reshape(state, [self.d,1]), [1, self.d] ) # all columns same
        diff = mat1 - mat2
        mat_alpha = tf.log( 1 + tf.exp( theta * (diff - self.shift))) # d x d
        # Compute Dirichlet PDF value for each row of action
        pdf = 1
        for row in range(self.d):
            dist = tf.distributions.Dirichlet(mat_alpha[row])
            pdf *= dist.prob(action[row])
        return pdf
        

    def calc_z_loop(self):
        """
        Calculates vector of z(traj_j) = [1/k sum_k q_k(traj_j)]^{-1}
        one element for each traj_j
        """
        # self.gen_actions is [num_sampled_trajectories*15, d, d]
        # reshape to [num_sampled_trajectories, 15, d, d]
        gen_actions_reshaped = tf.reshape(self.gen_actions, [self.num_sampled_trajectories,15,self.d,self.d])
        # self.gen_states is [num_sampled_trajectories*15, d]
        # reshape to [num_sampled_trajectories, 15, d]
        gen_states_reshaped = tf.reshape(self.gen_states, [self.num_sampled_trajectories,15,self.d])

        list_ops = []
        for j in range(self.num_sampled_trajectories):
            traj_actions = gen_actions_reshaped[j] # 15 x d x d
            traj_states = gen_states_reshaped[j] # 15 x d
            sum_q_over_k = 0
            for k in range(self.num_policies):
                theta = self.list_policies[k]
                # probability of start state Prob(s_1) is 1 / # starting samples
                # q_traj = p(s_1) prod_{t=1}^T q(a_t; s_t, theta)
                q_traj = 1.0 / self.num_start_samples
                # multiply pdf of each action across trajectory
                for t in range(15):
                    q_traj *= self.calc_pdf_action(theta, traj_actions[t], traj_states[t])
                sum_q_over_k += q_traj
            # z_j = [ 1/k sum_k q_k(traj_j) ]^{-1}
            list_ops.append( self.vec_z[j].assign( self.num_policies/sum_q_over_k ) )
        with tf.control_dependencies(list_ops):
            self.vec_z = tf.identity(self.vec_z)


    def calc_z(self):
        """
        Calculates vector of z(traj_j) = [1/k sum_k q_k(traj_j)]^{-1}
        one element for each traj_j
        """
        print("Inside calc_z")
        # self.gen_actions is [num_trajectories x 15, d, d]
        # Reshape actions to [num_trajectories, 1, time, d, d] with an extra dimension
        actions_reshaped = tf.reshape(self.gen_actions, [self.num_sampled_trajectories, 1, 15, self.d, self.d])
        # duplicate along the extra dimension <self.num_policies> times
        # [<num_trajectories>, <num_policies>, 15, d, d]
        self.actions_duplicated = tf.tile( actions_reshaped, [1,self.num_policies,1,1,1] )
        # replace all instances of 0 with 1e-6 to avoid raising 0 to negative power when
        # calculating pdf
        # self.actions_nozeros = tf.where( tf.equal(self.actions_duplicated,0), tf.ones_like(self.actions_duplicated)*1e-6, self.actions_duplicated)

        # Use self.gen_states to create alpha tensor
        # self.gen_states is [num_trajectories x 15 , d]
        # [<num_trajectories>, 15, d, d], all rows over 3rd dimension are same
        # Compare to mat1 in matrix case
        states_duplicated_rows = tf.tile( tf.reshape(self.gen_states, [self.num_sampled_trajectories, 15, 1, self.d]), [1,1,self.d,1] )
        # [<num_trajectories>, 15, d, d], all columns over 4th dimension are same
        # Compare to mat2 in matrix case
        states_duplicated_columns = tf.tile( tf.reshape(self.gen_states, [self.num_sampled_trajectories, 15, self.d, 1]), [1,1,1,self.d] )
        diff = states_duplicated_rows - states_duplicated_columns
        # Duplicate along one extra dimension <self.num_policies> times
        # [<num_trajectories>, <num_policies>, 15, d, d]
        diff_duplicated = tf.tile( tf.reshape(diff, [self.num_sampled_trajectories,1,15,self.d,self.d]), [1,self.num_policies,1,1,1] )
        # Weight by each policy's theta
        theta_duplicated = tf.reshape( tf.cast(self.list_policies, tf.float32), [1,self.num_policies, 1,1,1] )
        self.tensor_alpha = tf.log( 1 + tf.exp( tf.multiply((diff_duplicated - self.shift), theta_duplicated) ) ) # [<num_trajectories>, <num_policies>, 15, d, d]
        # Lower-bound alpha by 1, to avoid P_{ij}^{alpha - 1}
        # blowing up when P_{ij} is close to zero and alpha is less than 1
        self.tensor_alpha_lowerbound = tf.maximum( tf.scalar_mul(1+1e-6, tf.ones_like(self.tensor_alpha)), self.tensor_alpha )
        # Compute dirichlet for everything
        dist = tf.distributions.Dirichlet(self.tensor_alpha_lowerbound)
        # self.pdf = dist.prob(self.actions_nozeros) # [<num_trajectories>, <num_policies>, 15, d]
        self.pdf = dist.prob(self.actions_duplicated) # [<num_trajectories>, <num_policies>, 15, d]
        self.pdf = tf.cast(self.pdf, tf.float64)
        # Compute max value in pdf 
        # max_val = tf.reduce_max(self.pdf) # this does not work
        self.normalizer = tf.placeholder(dtype=tf.float64, shape=None)
        # Reduce magnitude
        self.pdf_normalized = self.pdf / self.normalizer

        # Now reduce everything
        # product over topic dimension to get q_k(P^t) from q_k(P^t_1)...q_k(P^t_d)
        self.reduce1 = tf.reduce_prod(self.pdf_normalized, axis=3) # [<num_trajectories>, <num_policies>, 15]
        # self.reduce1 = tf.reduce_prod(self.pdf, axis=3) # [<num_trajectories>, <num_policies>, 15]        
        # product over time to get q_k(tau_j) from q_k(P^1)...q_k(P^15)
        # NOTE: start state probability Pr(s_1) will be multiplied in below
        self.reduce2 = tf.reduce_prod(self.reduce1, axis=2) # [<num_trajectories>, <num_policies>]
        # sum over policy dimension to get sum_k q_k(tau_j)
        self.reduce3 = tf.reduce_sum(self.reduce2, axis=1) # [<num_trajectories>]
        # z_j = k / (sum_k q_k(tau_j)), along with start state probability
        self.vec_z = tf.cast(self.num_policies / (self.num_start_samples * self.reduce3), tf.float32)
        

    def create_training_method(self):
        """
        Defines loss, optimizer, and creates training operation.
        """
        print("Inside create_training_method")
        # first term = - 1/N sum_{traj_demo} r(traj_demo)
        # where N = number of sampled demo trajectories
        # just sum up r(s,a) over all (s_t,a_t) in each trajectory over all demo trajectories
        self.sum_demo_rewards = - 1.0/self.num_demo_samples * tf.reduce_sum(self.reward_demo)

        # second term = log(1/M sum_{traj_sample} z_{traj_sample} exp(r(traj_sample))
        # where z_{traj_sample} = [ 1/k sum_k q_k(traj_sample) ]^{-1}
        # where q_k is one policy and one traj is a sequence (s1,a1,...,s_t,a_t)
        # and M = total number of sampled trajectories (demo and generated)
        # reshape into matrix of r(s_{it}, a_{it}) for i = 1...M and t = 1...15
        gen_rewards_reshaped = tf.reshape( self.reward_gen, [self.num_sampled_trajectories, 15] )
        # sum over time to get vector of r(traj_sample) for each traj_sample
        gen_rewards_per_traj = tf.reduce_sum( gen_rewards_reshaped, axis=1 )
        # exponentiate to get exp(r(traj_sample))
        gen_rewards_exp = tf.exp(gen_rewards_per_traj) # [num_sampled_trajectories]

        # calculate vector of z_{traj_sample} = [ 1/k sum_k q_k(traj_sample) ]^{-1}
        # self.calc_z()
        # second_term = tf.log( 1.0/self.num_sampled_trajectories * tf.reduce_sum( self.vec_z * gen_rewards_exp) )
        self.second_term = tf.log( 1.0 / self.num_sampled_trajectories * tf.reduce_sum( gen_rewards_exp) ) 

        # compute loss = negative log likelihood
        if self.reg == 'l1l2' or self.reg == 'dropout_l1l2':
            reg_losses = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
            self.loss = self.sum_demo_rewards + self.second_term + sum(reg_losses)
        else:
            self.loss = self.sum_demo_rewards + self.second_term

        if self.summarize:
            tf.summary.scalar('loss', self.loss)
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_reward)
        self.r_train_op = self.optimizer.minimize(self.loss)

        if self.summarize:
            for v in tf.trainable_variables():
                tf.summary.histogram(v.op.name, v)
            grads = self.optimizer.compute_gradients(self.loss, tf.trainable_variables())
            for grad, var in grads:
                if grad is not None:
                    tf.summary.histogram(var.op.name + '/gradients', grad)


    def init_w(self, d):
        """
        Input:
        d - number of topics

        Feature vector is 
        [1, pi_1,...,pi_d, pi_1*pi_1,...,pi_1*pi_d, pi_2*pi_2,...,pi_2*pi_d, ...... , pi_d*pi_d]
        Initialize weight vector for value function approximation
        Need to decide whether to include the null topic
        """
        num_features = int((d+1)*d / 2 + d + 1)
        return np.random.rand(num_features, 1)


    def init_pi0(self, path_to_dir, verbose=0):
        """
        Generates the collection of initial population distributions.
        This collection will be sampled to get the start state for each training episode
        Assumes that each file in directory has rows of the format:
        pi^0_1 pi^0_2 ... pi^0_d
        ...
        pi^d_1 pi^d_2 ... pi^d_d
        where d is a fixed constant across all files
        """
        # will be list of lists
        list_pi0 = []
        num_files = len(os.listdir(path_to_dir))

        for num_day in range(1, 1+num_files):
            filename = "trend_distribution_day%d.csv" % num_day
            path_to_file = path_to_dir + '/' + filename
            with open(path_to_file, 'r') as f:
                list_lines = f.readlines()
            # Take first line, split by ' ', map to float, convert to list and append to list_pi0
            list_pi0.append( list(map(float, list_lines[0].strip().split(' ')))[0:self.d] )
            if verbose:
                print(filename)
            
        num_rows = len(list_pi0)
        num_cols = len(list_pi0[0])

        # Convert to np matrix
        self.mat_pi0 = np.zeros([num_rows, num_cols])
        for i in range(len(list_pi0)):
            self.mat_pi0[i] = list_pi0[i]
        

    def sample_action(self, pi):
        """
        Samples from product of d d-dimensional Dirichlet distributions
        Input:
        pi - row vector
        Returns an entire transition probability matrix
        """
        # Construct all alphas
        self.mat_alpha = np.zeros([self.d, self.d])

        # temp_{ij} = pi_j - pi_i
        # alpha^i_j = ln ( 1 + exp[ theta ( (pi_j - pi_i) - shift ) ] )
        mat1 = np.repeat(pi.reshape(1, self.d), self.d, 0) # all rows same
        mat2 = np.repeat(pi.reshape(self.d, 1), self.d, 1) # all columns same
        temp = mat1 - mat2
        self.mat_alpha = np.log( 1 + np.exp( self.theta * (temp - self.shift)))

        # Sample matrix P from Dirichlet
        P = np.zeros([self.d, self.d])
        for i in range(self.d):
            # Get y^i_1, ... y^i_d
            # Using the vector as input to shape reduces runtime by 5s
            try:
                y = np.random.gamma(shape=self.mat_alpha[i,:]*self.alpha_scale, scale=1)
            except ValueError:
                print("ValueError!")
                print(pi)
                print(self.mat_alpha)
                print(self.mat_alpha[i,:]*self.alpha_scale)
            # replace zeros with dummy value
            y[y == 0] = 1e-20
            total = np.sum(y)
            # Store into i-th row of matrix P
            try:
                P[i] = y / total
            except Warning:
                P[i] = y / total
                print(y, total)

        return P


    def calc_features(self, pi):
        """
        Input:
        pi - population distribution as a row vector

        Returns varphi(pi) as a row vector
        """        
        # generate pairs of (pi_i, pi_j) for all i, for all j >= i
        list_tuples = list(itertools.combinations_with_replacement(pi, 2))
        # calculate products
        list_features = []
        for idx in range(len(list_tuples)):
            pair = list_tuples[idx]
            list_features.append(pair[0] * pair[1])
        # append first-order feature
        list_features = list_features + list(pi)
        # append bias
        list_features.append(1)

        return np.array(list_features)


    def calc_alpha_deriv(self, pi):
        """
        Calculates derivative of alpha^i_j = ln(1 + exp(theta((pi_j - pi_i) - s)))
        pi - row vector
        """
        # matrix of derivatives
        self.mat_alpha_deriv = np.zeros([self.d, self.d])
        
        mat1 = np.repeat(pi.reshape(1, self.d), self.d, 0) # all rows same
        mat2 = np.repeat(pi.reshape(self.d, 1), self.d, 1) # all columns same
        diff = mat1 - mat2

        # d(alpha^i_j)/d(theta) = \frac{ pi_j - pi_i - shift } { 1 + exp( -theta*(pi_j - pi_i - shift) ) }
        numerator = diff - self.shift
        denominator = 1 + np.exp( (-self.theta) * numerator)
        self.mat_alpha_deriv = numerator / denominator


    def calc_gradient_vectorized(self, P, pi):
        """
        Input:
        P - transition probability matrix
        pi - population distribution as a row vector

        Calculates \nabla_{theta} log (F(P, pi, theta))
        where F is the product of d d-dimensional Dirichlet distributions
        tensor_phi and mat_alpha are global variables computed in sample_action()

        This version is ~3 times faster than the non-vectorized version
        """
        # compute self.mat_alpha_deriv for use in gradient
        self.calc_alpha_deriv(pi)

        # (i,j) element of mat1 is psi(alpha^i_j)
        mat1 = special.digamma(self.mat_alpha)
        # Each row of mat2 has same value along the row
        # each element in row i is psi(\sum_j alpha^i_j)
        mat2 = special.digamma( np.ones([self.d, self.d]) * np.sum(self.mat_alpha, axis=1, keepdims=True) )
        # (i,j) element of mat3 is ln(P_{ij})
        P[P==0] = 1e-100
        try:
            mat3 = np.log(P)
        except Warning:
            print(P)
            print(np.where( P==0 )[0])
            mat3 = np.log(P)

        # Expression is
        # nabla_theta log(F) = \sum_i \sum_j (-psi(alpha^i_j) + psi(\sum_j alpha^i_j) + ln(P_{ij})) d(alpha^i_j)/d(theta)
        gradient = np.sum( (-mat1 + mat2 + mat3) * self.mat_alpha_deriv )

        return gradient


    def train_log(self, vector, filename, str_format):
        f = open(filename, 'a')
        vector.tofile(f, sep=',', format=str_format)
        f.write("\n")
        f.close()
    

    def train(self, max_episodes=4000, stop_criteria=0.01, gamma=1, constant=False, lr_critic=0.1, lr_actor=0.001, consecutive=100, file_theta='results/theta.csv', file_pi='results/pi.csv', file_reward='results/reward.csv', write_file=0, write_all=0):
        """
        Main actor-critic training procedure that improves theta and w

        Input:
        max_episodes - each episode is 16 steps (9am to 12midnight)
        gamma - temporal discount
        constant - if True, then does not decrease learning rates
        lr_critic - learning rate for value function parameter update
        lr_actor - learning rate for policy parameter update
        consecutive - number of consecutive episodes between report of average reward
        """
        print("----- Starting train -----")
        list_reward = []
        prev_theta = self.theta
        for episode in range(1, max_episodes+1):
            # print("forward episode ", episode)
            if write_all:
                with open('temp.csv', 'a') as f:
                    f.write('Episode %d \n\n' % episode)
            # Sample starting pi^0 from mat_pi0
            idx_row = np.random.randint(self.num_start_samples)
            pi = self.mat_pi0[idx_row, :] # row vector

            discount = 1
            total_reward = 0
            num_steps = 0

            # Stop after finishing the iteration when num_steps=15, because
            # at that point pi_next = the predicted distribution at midnight
            while num_steps < 15:
                num_steps += 1

                # Sample action
                P = self.sample_action(pi)

                if write_all:
                    with open('temp.csv', 'ab') as f:
                        np.savetxt(f, np.array(['num_steps = %d' % num_steps]), fmt='%s')
                        np.savetxt(f, np.array(['distribution']), fmt='%s')
                        np.savetxt(f, pi.reshape(1, self.d), delimiter=',', fmt='%.6f')
                        np.savetxt(f, np.array(['Action']), fmt='%s')
                        np.savetxt(f, P, delimiter=',', fmt='%.3f')
            
                # Take action, get pi^{n+1} = P^T pi
                pi_next = np.transpose(P).dot(pi)

                # Calculate reward
                # reward = self.calc_reward(P, pi, self.d)
                reward = self.sess.run( self.reward_gen, feed_dict={self.gen_states:[pi], self.gen_actions:[P]} )
                if np.isnan(reward) or reward == np.inf or reward == -np.inf:
                    print(reward)
                
                # Calculate TD error
                vec_features_next = self.calc_features(pi_next)
                vec_features = self.calc_features(pi)
                # TD error = r + gamma * v(s'; w) - v(s; w)
                delta = reward + discount*(vec_features_next.dot(self.w)) - (vec_features.dot(self.w))

                # Update value function parameter
                # w <- w + alpha * TD error * feature vector
                # still a column vector
                length = len(vec_features)
                if constant:
                    self.w = self.w + lr_critic * delta * vec_features.reshape(length,1)
                else:
                    self.w = self.w + (lr_critic/(episode+1)) * delta * vec_features.reshape(length,1) #here

                # Update policy parameter
                # theta <- theta + beta * grad(log(F)) * TD error
                gradient = self.calc_gradient_vectorized(P, pi)
                if constant:
                    self.theta = self.theta + lr_actor * delta * gradient
                else:
                    self.theta = self.theta + (lr_actor/((episode+1)*np.log(np.log(episode+20)))) * delta * gradient #here

                discount = discount * gamma
                pi = pi_next
                total_reward += reward

            list_reward.append(total_reward)

            if (episode % consecutive == 0):
                print("Theta\n", self.theta)
                print("pi\n", pi)
                reward_avg = sum(list_reward)/consecutive
                print("Average reward during previous %d episodes: " % consecutive, str(reward_avg))
                list_reward = []
                if write_file:
                    self.train_log(self.theta, file_theta, "%.5e")
                    self.train_log(pi, file_pi, "%.3e")
                    self.train_log(np.array([reward_avg]), file_reward, "%.3e")
            if stop_criteria != -1 and abs(self.theta - prev_theta) < stop_criteria:
                break
            prev_theta = self.theta

        # record this policy
        self.list_policies = (self.list_policies + [self.theta])[1:]
        print("----- Exiting train at episode %d with theta %f -----" % (episode, self.theta))


    def generate_trajectories(self, n):
        """
        Use the current policy self.theta to generate trajectories
        n - number of trajectories to generate

        Return: list of generated trajectories
        """
        print("Inside generate_trajectories")
        # Will be list of lists of tuples of form (state, action)
        list_generated = []
        max_hour = 16

        for idx_traj in range(n):
            trajectory = []
            # Sample start state
            idx_row = np.random.randint(self.num_start_samples)
            pi = self.mat_pi0[idx_row, :] # row vector

            # Generate trajectory, i.e. a list of state-action pairs
            hour = 1
            while hour < max_hour:
                P = self.sample_action(pi)
                trajectory.append( (pi, P) )
                pi = np.transpose(P).dot(pi)
                hour += 1
            list_generated.append( trajectory )

        return list_generated


    def debug(self, feed_dict):
        self.tensor_alpha_val = self.sess.run(self.tensor_alpha, feed_dict=feed_dict)
        self.tensor_alpha_lowerbound_val = self.sess.run(self.tensor_alpha_lowerbound, feed_dict=feed_dict)

        self.pdf_val = self.sess.run(self.pdf, feed_dict=feed_dict)
        print("pdf max val = ", np.max(self.pdf_val))
        print("pdf min val = ", np.min(self.pdf_val))
        pdf_max_along_rows = np.max(self.pdf_val, axis=3)
        pdf_min_along_rows = np.min(self.pdf_val, axis=3)
        pdf_diff = pdf_max_along_rows - pdf_min_along_rows
        pdf_max_diff = np.max(pdf_diff)
        print("pdf max over everything of difference between max and min val for a single P matrix ", pdf_max_diff)

        self.pdf_normalized_val = self.sess.run(self.pdf_normalized, feed_dict=feed_dict)
        print("pdf_normalized[0,0,0]")
        print(self.pdf_normalized_val[0,0,0])

        self.reduce1_val = self.sess.run(self.reduce1, feed_dict=feed_dict)
        print("reduce1[0,0]")
        print(self.reduce1_val[0,0])
    
        self.reduce2_val = self.sess.run(self.reduce2, feed_dict=feed_dict)
        print("reduce2[0]")
        print(self.reduce2_val[0])
    
        self.reduce3_val = self.sess.run(self.reduce3, feed_dict=feed_dict)
        print("reduce3")
        print(self.reduce3_val)
    
        self.vec_z_val = self.sess.run(self.vec_z, feed_dict=feed_dict)
        print("vec_z")
        print(self.vec_z_val)


    def update_reward(self, summary=False, iteration=0):
        """
        Improvement of reward function via gradient descent

        summary - if True, then run tf.summary ops
        iteration - global iteration count for number of reward updates so far
        """
        # print("In update_reward")
        # Sample demonstrations from self.list_demonstrations,
        # which is list of lists of tuples
        if len(self.list_demonstrations) >= self.num_demo_samples:
            demo_sampled = random.sample(self.list_demonstrations, self.num_demo_samples)
        else:
            demo_sampled = self.list_demonstrations[:]
        # Separate into actions and states to calculate self.reward_demo
        demo_states = [pair[0] for traj in demo_sampled for pair in traj]
        demo_actions = [pair[1] for traj in demo_sampled for pair in traj]

        # Sample generated trajectories from self.list_generated
        if len(self.list_generated) >= self.num_gen_samples:
            gen_sampled = random.sample(self.list_generated, self.num_gen_samples)
        else:
            gen_sampled = self.list_generated[:]
        # Separate into actions and states to calculate self.reward_gen
        gen_states = [pair[0] for traj in gen_sampled for pair in traj]
        gen_actions = [pair[1] for traj in gen_sampled for pair in traj]

        # Combine
        # gen_states = gen_states + demo_states
        # gen_actions = gen_actions + demo_actions

        # feed_dict = {self.demo_states:demo_states, self.demo_actions:demo_actions, self.gen_states:gen_states, self.gen_actions:gen_actions, self.normalizer:self.c}
        feed_dict = {self.demo_states:demo_states, self.demo_actions:demo_actions, self.gen_states:gen_states, self.gen_actions:gen_actions}

        # debugging
        # self.debug(feed_dict)

        # Execute gradient descent
        if summary and self.summarize:
            summary, _, self.loss_val, self.first_term_val, self.second_term_val = self.sess.run([self.merged, self.r_train_op, self.loss, self.sum_demo_rewards, self.second_term], feed_dict=feed_dict)
            self.train_writer.add_summary(summary, iteration)
        else:
            _, self.loss_val, self.first_term_val, self.second_term_val = self.sess.run([self.r_train_op, self.loss, self.sum_demo_rewards, self.second_term], feed_dict=feed_dict)


    def reward_iteration(self, max_iterations=500, stop_criteria=0.01, iter_check=10):
        """
        max_iterations - obvious
        stop_critiera - stop when absolute difference between reward_demo_avg and previous value is less than this value
        iter_check - number of iterations between each check of avg reward on demo and generated samples

        Return True if only ran <= 2*iter_check iterations, which indicates
        reward network has stabilized
        Else return False
        """
        prev_reward_demo_avg = -100
        reward_demo_avg = 0
        reward_gen_avg = 0

        print("----- Starting reward_iteration -----")
        for it in range(1, max_iterations+1):

            self.reward_update_count += 1

            if it % iter_check != 0:
                self.update_reward(summary=False)
            else:
                print("Reward iteration %d" % it)
                self.update_reward(summary=False, iteration=self.reward_update_count)

                demo_states = [pair[0] for pair in self.list_eval_demo_transitions]
                demo_actions = [pair[1] for pair in self.list_eval_demo_transitions]
                num_test_demo = len(self.list_eval_demo_transitions)
                gen_states = [pair[0] for pair in self.list_eval_gen_transitions]
                gen_actions = [pair[1] for pair in self.list_eval_gen_transitions]
                num_test_gen = len(self.list_eval_gen_transitions)
                feed_dict = {self.demo_states:demo_states, self.demo_actions:demo_actions, self.gen_states:gen_states, self.gen_actions:gen_actions}
                
                reward_demo_val, reward_gen_val = self.sess.run([self.reward_demo, self.reward_gen], feed_dict=feed_dict )
                # average reward across state-action pairs
                reward_demo_avg = np.sum(reward_demo_val) / num_test_demo
                reward_gen_avg = np.sum(reward_gen_val) / num_test_gen
                print("Reward demo avg %f | Reward gen avg %f" % (reward_demo_avg, reward_gen_avg))
                print("First %f | Second %f | Loss %f" % (self.first_term_val, self.second_term_val, self.loss_val))

                if np.isnan(reward_demo_avg) or np.isnan(reward_gen_avg):
                    break
                with open("results/reward_training.csv", 'a') as f:
                    f.write("%f,%f\n" % (reward_demo_avg, reward_gen_avg))
                if stop_criteria != -1 and abs(reward_demo_avg - prev_reward_demo_avg) < stop_criteria:
                    break
                prev_reward_demo_avg = reward_demo_avg

        print("----- Exiting reward_iteration at iter %d -----" % it)


    def outerloop(self, num_iterations=20, num_gen_from_policy=5, max_reward_iterations=100, max_forward_episodes=200, gamma=1, constant=False, lr_critic=0.1, lr_actor=0.001):
        """
        Outer-most loop that calls functions to update reward function
        and solve the forward problem

        num_iterations - number of outerloop iterations
        num_gen_from_policy - number of trajectories to generate using policy at each step
        max_reward_iterations - maximum number of reward iterations to do
        max_forward_episodes - number of episodes to run when solving forward problem
        gamma - temporal discount
        constant - if True, then does not decrease learning rates
        lr_critic - learning rate for value function parameter update
        lr_actor - learning rate for policy parameter update
        """

        # At the beginning, generate trajectories from initial policies, all of which
        # are the same. This is meant to populate the data for use in reward learning,
        # before accumulating <num_policies> different policies from forward passes
        self.list_generated = self.generate_trajectories(num_gen_from_policy * self.num_policies)
        # Initialize reward update counter for writing to tensorboard
        self.reward_update_count = 0
        with open("results/reward_training.csv", 'w') as f:
            f.write("reward_demo_avg,reward_gen_avg\n")

        for it in range(num_iterations):
            print("########## Outerloop iteration %d ##########" % it)
            # Generate samples D_traj from current policy
            list_generated = self.generate_trajectories(num_gen_from_policy)

            # D_samp <- D_samp union D_traj
            self.list_generated = self.list_generated + list_generated
            # kick out trajectories generated from the earliest policy
            self.list_generated = self.list_generated[num_gen_from_policy: ]

            # Get a list of transitions from generated trajectories, for evaluating reward
            self.list_eval_gen_transitions = [pair for traj in self.list_generated for pair in traj]
            # self.list_eval_gen_transitions = self.get_eval_transitions(self.list_generated)

            # Update reward function
            self.reward_iteration(max_iterations=max_reward_iterations, stop_criteria=0.0001, iter_check=10)

            # Solve forward problem
            self.theta = self.theta_initial
            self.train(max_forward_episodes, -1, gamma, constant, lr_critic, lr_actor, consecutive=100, file_theta='results/theta.csv', file_pi='results/pi.csv', file_reward='results/reward.csv', write_file=1, write_all=0)
            print("\n")

        # Save reward network
        print("Saving network")
        self.saver.save(self.sess, "log/model_%s_%d_%d.ckpt" % (self.reg, self.n_fc3, self.n_fc4))

        # Solve forward problem completely
        print("********** Final forward training **********")
        self.theta = self.theta_initial
        self.train(2000, -1, gamma, constant, lr_critic, lr_actor, consecutive=100, file_theta='results/theta.csv', file_pi='results/pi.csv', file_reward='results/reward.csv', write_file=1, write_all=0)
        return self.theta


# ---------------- End training code ---------------- #

# ---------------- Evaluation code ------------------ #

    def test_convergence(self, num_iterations=500, num_gen_from_policy=5, iter_check=10, filename='reward_convergence.csv'):
        """
        Trains reward network using batches from demonstration set and 
        fixed generated set, using fixed policy
        Checks whether reward value of a demonstration trajectory converges.

        num_iterations - number of minibatch gradient descent iterations 
        num_gen_from_policy - number of trajectories to generate using fixed policy
        iter_check - check reward value on selected demo trajectory after every <iter_check> iterations
        """
        self.list_generated = self.generate_trajectories(num_gen_from_policy * self.num_policies)

        # Get a list of transitions from generated trajectories, for testing reward function
        self.list_eval_gen_transitions = [pair for traj in self.list_generated for pair in traj]
        # self.list_eval_gen_transitions = self.get_eval_transitions(self.list_generated)
        
        with open("results/" + filename, 'w') as f:
            f.write("iteration,reward_demo_avg,reward_gen_avg\n")

        for it in range(1, num_iterations+1):
            
            if it % iter_check != 0:
                self.update_reward(summary=False)
            else:
                print("Iteration %d" % it)
                self.update_reward(summary=False, iteration=it)
                
                demo_states = [pair[0] for pair in self.list_eval_demo_transitions]
                demo_actions = [pair[1] for pair in self.list_eval_demo_transitions]
                num_test_demo = len(self.list_eval_demo_transitions)
                gen_states = [pair[0] for pair in self.list_eval_gen_transitions]
                gen_actions = [pair[1] for pair in self.list_eval_gen_transitions]
                num_test_gen = len(self.list_eval_gen_transitions)
                feed_dict = {self.demo_states:demo_states, self.demo_actions:demo_actions, self.gen_states:gen_states, self.gen_actions:gen_actions}
                
                reward_demo_val, reward_gen_val = self.sess.run([self.reward_demo, self.reward_gen], feed_dict=feed_dict )
                # average reward across all transitions
                reward_demo_avg = np.sum(reward_demo_val) / num_test_demo
                reward_gen_avg = np.sum(reward_gen_val) / num_test_gen
                print("Reward demo avg %f | Reward gen avg %f" % (reward_demo_avg, reward_gen_avg))
                print("First %f | Second %f | Loss %f" % (self.first_term_val, self.second_term_val, self.loss_val))
                with open("results/" + filename, 'a') as f:
                    f.write("%d,%f,%f\n" % (it, reward_demo_avg, reward_gen_avg))
                if np.isnan(reward_demo_avg) or np.isnan(reward_gen_avg):
                    break
                

    def test_reward_network(self):
        """
        Given a fixed reward network, evaluate
        average reward over all transitions in training demonstrations
        average reward over all transitions in test (or validation) demonstrations
        average reward over all transitions in generated trajectories
        """

        # Evaluate on demonstration training set and generated set
        num_demos = len(self.list_demonstrations)
        self.list_generated = self.generate_trajectories(num_demos)
        gen_states = [pair[0] for traj in self.list_generated for pair in traj]
        gen_actions = [pair[1] for traj in self.list_generated for pair in traj]
        num_test_gen = len(gen_states)

        demo_states = [pair[0] for traj in self.list_demonstrations for pair in traj]
        demo_actions = [pair[1] for traj in self.list_demonstrations for pair in traj]
        num_test_demo = len(demo_states)

        feed_dict = {self.demo_states:demo_states, self.demo_actions:demo_actions, self.gen_states:gen_states, self.gen_actions:gen_actions}
        reward_demo_train, reward_gen_val = self.sess.run([self.reward_demo, self.reward_gen], feed_dict=feed_dict )
        reward_demo_avg_train = np.sum(reward_demo_train) / num_test_demo
        reward_gen_avg = np.sum(reward_gen_val) / num_test_gen

        # Evaluate on demonstration validation or test set
        demo_states = [pair[0] for traj in self.list_demonstrations_test for pair in traj]
        demo_actions = [pair[1] for traj in self.list_demonstrations_test for pair in traj]
        num_test_demo = len(demo_states)

        feed_dict = {self.demo_states:demo_states, self.demo_actions:demo_actions}
        reward_demo_test = self.sess.run(self.reward_demo, feed_dict=feed_dict )
        reward_demo_avg_test = np.sum(reward_demo_test) / num_test_demo

        print("Avg reward demo train %f | Avg reward demo test %f | Avg reward gen %f" % (reward_demo_avg_train, reward_demo_avg_test, reward_gen_avg))

        return reward_demo_avg_train, reward_demo_avg_test, reward_gen_avg


    def plot_reward_distribution(self, theta_good=8.06, xmin=-0.1, xmax=0.4, filename='reward_distribution.pdf'):
        """
        Generate three histograms
        1. distribution of reward for demo transitions
        2. distribution of reward for demo test transitions
        2. distribution of reward for good generated transitions

        theta_good - learned theta for generating good transitions
        xmin, xmax, ymin, ymax - axis boundaries
        """

        # Get rewards on demo transitions
        demo_states = [pair[0] for traj in self.list_demonstrations for pair in traj]
        demo_actions = [pair[1] for traj in self.list_demonstrations for pair in traj]
        feed_dict = {self.demo_states:demo_states, self.demo_actions:demo_actions}
        reward_demo_val = self.sess.run(self.reward_demo, feed_dict=feed_dict)

        # Rewards on demo test transitions
        demo_test_states = [pair[0] for traj in self.list_demonstrations_test for pair in traj]
        demo_test_actions = [pair[1] for traj in self.list_demonstrations_test for pair in traj]
        feed_dict = {self.demo_states:demo_test_states, self.demo_actions:demo_test_actions}
        reward_demo_test_val = self.sess.run(self.reward_demo, feed_dict=feed_dict)

        # Generate list of trajectories using good policy
        num_demos = len(self.list_demonstrations)
        self.theta = theta_good
        list_generated_good = self.generate_trajectories(num_demos)
        gen_states = [pair[0] for traj in list_generated_good for pair in traj]
        gen_actions = [pair[1] for traj in list_generated_good for pair in traj]
        feed_dict = {self.gen_states:gen_states, self.gen_actions:gen_actions}
        reward_gen_good = self.sess.run(self.reward_gen, feed_dict=feed_dict)

        fig = plt.figure(1)

        density_demo = gaussian_kde(reward_demo_val.flatten())
        xs = np.linspace(xmin, xmax, 200)
        # density_demo.covariance_factor = lambda : .25
        # density_demo._compute_covariance()
        plt.plot(xs, density_demo(xs), label='Demo (train)', color='g')

        density_demo_test = gaussian_kde(reward_demo_test_val.flatten())
        plt.plot(xs, density_demo_test(xs), label='Demo (test)', color='r')

        density_gen = gaussian_kde(reward_gen_good.flatten())
        plt.plot(xs, density_gen(xs), label='Generated', color='b')        

        plt.ylabel('Density')
        plt.xlabel('Reward')
        # plt.axis([xmin, xmax, ymin, ymax1])
        plt.title('Distribution of reward for demo and generated transitions')
        plt.legend(loc='best', fontsize=14)        
        axes = plt.gca()
        for item in ([axes.title, axes.yaxis.label, axes.xaxis.label]):
            item.set_fontsize(14)

        plt.tight_layout()
        pp = PdfPages('plots_irl/'+filename)
        pp.savefig(fig, bbox_inches='tight')
        pp.close()

        dist_demo, _, _ = plt.hist(reward_demo_val, bins=30, normed=0, facecolor='g')
        dist_demo_test, _, _ = plt.hist(reward_demo_test_val, bins=30, normed=0, facecolor='r')
        dist_gen, _, _ = plt.hist(reward_gen_good, bins=30, normed=0, facecolor='b')

        print("JSD between demo and demo_test", self.JSD(dist_demo, dist_demo_test))
        print("JSD between demo and gen", self.JSD(dist_demo, dist_gen))


    def plot_reward_histogram(self, theta_good=8.06, xmin=-0.1, xmax=0.4, ymin=0, ymax1=50, ymax2=25, ymax3=50, x_text=0.2, y_text1=35, y_text2=20, y_text3=35, filename='reward_histogram.pdf'):
        """
        Generate three histograms
        1. reward for demo transitions
        2. reward for demo test transitions
        3. reward for good generated transitions

        theta_good - learned theta for generating good transitions
        xmin, xmax, ymin, ymax - axis boundaries
        """

        # Get rewards on demo transitions
        demo_states = [pair[0] for traj in self.list_demonstrations for pair in traj]
        demo_actions = [pair[1] for traj in self.list_demonstrations for pair in traj]
        feed_dict = {self.demo_states:demo_states, self.demo_actions:demo_actions}
        reward_demo_val = self.sess.run(self.reward_demo, feed_dict=feed_dict)

        # Rewards on demo test transitions
        demo_test_states = [pair[0] for traj in self.list_demonstrations_test for pair in traj]
        demo_test_actions = [pair[1] for traj in self.list_demonstrations_test for pair in traj]
        feed_dict = {self.demo_states:demo_test_states, self.demo_actions:demo_test_actions}
        reward_demo_test_val = self.sess.run(self.reward_demo, feed_dict=feed_dict)

        # Generate list of trajectories using good policy
        num_demos = len(self.list_demonstrations)
        self.theta = theta_good
        list_generated_good = self.generate_trajectories(num_demos)
        gen_states = [pair[0] for traj in list_generated_good for pair in traj]
        gen_actions = [pair[1] for traj in list_generated_good for pair in traj]
        feed_dict = {self.gen_states:gen_states, self.gen_actions:gen_actions}
        reward_gen_good = self.sess.run(self.reward_gen, feed_dict=feed_dict)

        fig = plt.figure(1)

        plt.subplot(311)
        dist_demo, _, _ = plt.hist(reward_demo_val, bins=30, normed=0, facecolor='g')
        plt.ylabel('Count')
        plt.axis([xmin, xmax, ymin, ymax1])
        plt.title('Histogram of reward for demo and generated transitions')
        plt.text(x_text, y_text1,  r'Demo (train)')
        axes = plt.gca()
        for item in ([axes.title, axes.yaxis.label]):
            item.set_fontsize(14)
        
        plt.subplot(312)
        dist_demo_test, _, _ = plt.hist(reward_demo_test_val, bins=30, normed=0, facecolor='r')
        plt.ylabel('Count')
        plt.axis([xmin, xmax, ymin, ymax2])
        plt.text(x_text, y_text2,  r'Demo (test)')
        axes = plt.gca()
        axes.yaxis.label.set_fontsize(14)
        
        plt.subplot(313)
        dist_gen, _, _ = plt.hist(reward_gen_good, bins=30, normed=0, facecolor='b')
        plt.ylabel('Count')
        plt.axis([xmin, xmax, ymin, ymax3])
        plt.text(x_text, y_text3, r'Generated')
        plt.xlabel('Reward')
        axes = plt.gca()
        for item in ([axes.yaxis.label, axes.xaxis.label]):
            item.set_fontsize(14)

        plt.tight_layout()
        pp = PdfPages('plots_irl/'+filename)
        pp.savefig(fig, bbox_inches='tight')
        pp.close()

        print("JSD between demo and demo_test", self.JSD(dist_demo, dist_demo_test))
        print("JSD between demo and gen", self.JSD(dist_demo, dist_gen))        


    def plot_action_heatmap(self, theta_good, filename='action_heatmap.pdf'):
        """
        Does not require reward network. Generate three heatmaps:
        1. distribution of averaged demo actions
        2. distribution of averaged good generated actions
        3. distribution of averaged bad generated actions

        theta_good - learned theta for generating good transitions
        theta_bad - some random bad theta for generating bad transitions
        """
        # Get demo actions
        demo_actions = [pair[1] for traj in self.list_demonstrations for pair in traj]
        demo_actions_arr = np.asarray(demo_actions)
        demo_avg = np.mean(demo_actions_arr, axis=0)

        # Generate list of trajectories using good policy
        num_demos = len(self.list_demonstrations)
        self.theta = theta_good
        list_generated_good = self.generate_trajectories(num_demos)
        gen_actions_good = [pair[1] for traj in list_generated_good for pair in traj]
        gen_actions_good_arr = np.asarray(gen_actions_good)
        gen_good_avg = np.mean(gen_actions_good_arr, axis=0)

        diff = np.abs(demo_avg - gen_good_avg)

        fig, axes = plt.subplots(nrows=1, ncols=2)

        ax0 = axes[0]
        im = ax0.imshow(demo_avg, cmap='hot', vmin=0, vmax=1)
        ax0.set_title('Demonstration actions')
        ax0.title.set_fontsize(14)
        major_ticks = np.arange(0, 15, 5) 
        ax0.set_xticks(major_ticks)
        ax0.set_yticks(major_ticks)

        ax1 =  axes[1]
        im = ax1.imshow(diff, cmap='hot', vmin=0, vmax=1)
        ax1.set_title('Difference')
        ax1.title.set_fontsize(14)
        ax1.set_xticks(major_ticks)
        ax1.set_yticks(major_ticks)        

        fig.subplots_adjust(right=0.8)
        cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.4])
        fig.colorbar(im, cax=cbar_ax)
        
        pp = PdfPages('plots_irl/'+filename)
        pp.savefig(fig, bbox_inches='tight')
        pp.close()        


    def JSD(self, P, Q):
        """
        Arguments:
        P,Q - discrete probability distribution
        
        Return:
        Jensen-Shannon divergence
        """

        # Replace all zeros by 1e-100
        P[P==0] = 1e-100
        Q[Q==0] = 1e-100

        P_normed = P / norm(P, ord=1)
        Q_normed = Q / norm(Q, ord=1)
        M = 0.5 * (P + Q)

        return 0.5 * (entropy(P,M) + entropy(Q,M))


    def generate_trajectory(self, pi0, total_hours):
        """
        Argument:
        pi0 - initial population distribution (included in output)
        total_hours - number of hours to generate (including first and last hour)

        Return:
        Matrix, each row is the distribution at a discrete time step,
        from pi^0 to pi^N
        """

        pi = pi0
        # Initialize matrix to store trajectory
        # total_steps rows by d columns
        mat_trajectory = np.zeros([total_hours, self.d])
        # Store initial distribution
        mat_trajectory[0] = pi
        hour = 1

        while hour < total_hours:
            P = self.sample_action(pi)
            pi_next = np.transpose(P).dot(pi)
            mat_trajectory[hour] = pi_next
            pi = pi_next
            hour += 1

        return mat_trajectory


    def evaluate(self, theta=8.86349, shift=0.5, alpha_scale=1e4, d=15, episode_length=16, indir='test_normalized_round2', outfile='eval_mfg_round2/validation.csv', write_header=0):
        """
        Main evaluation function

        Argument:
        theta - value to use for the fixed policy
        indir - directory containing the test dataset

        """
        # Fix policy by setting parameter
        self.theta = theta
        self.shift = shift
        self.alpha_scale = alpha_scale
        self.d = d
        
        path_to_dir = os.getcwd() + '/' + indir
        num_test_trajectories = len(os.listdir(path_to_dir))
        array_l1_final = np.zeros(num_test_trajectories)
        array_l1_mean = np.zeros(num_test_trajectories)
        array_JSD_final = np.zeros(num_test_trajectories)
        array_JSD_mean = np.zeros(num_test_trajectories)
        
        idx = 0
        # For each file in test_normalized
        for filename in os.listdir(path_to_dir):
            path_to_file = path_to_dir + '/' + filename

            with open(path_to_file, 'r') as f:
                mat_empirical = np.loadtxt(f, delimiter=' ')
                mat_empirical = mat_empirical[:, 0:self.d]

            # Read initial distribution pi0
            pi0 = mat_empirical[0]

            # Generate entire trajectory using policy
            mat_trajectory = self.generate_trajectory(pi0, episode_length)

            # L1 norm of difference between generated and empirical final distribution pi^N
            l1_final = norm(mat_trajectory[-1] - mat_empirical[-1], ord=1)
            array_l1_final[idx] = l1_final

            # L1 norm of difference between generated distribution and empirical distribution, averaged across all time steps
            diff = mat_empirical - mat_trajectory
            l1_mean = np.mean(np.apply_along_axis(lambda row: norm(row, ord=1), 1, diff))
            array_l1_mean[idx] = l1_mean

            # JS divergence between final distributions
            JSD_final = self.JSD(mat_trajectory[-1], mat_empirical[-1])
            array_JSD_final[idx] = JSD_final

            # Average JS divergence across all time steps
            JSD_mean = 0
            for idx2 in range(episode_length):
                JSD_mean += self.JSD(mat_empirical[idx2], mat_trajectory[idx2])
                
            JSD_mean = JSD_mean / episode_length
            array_JSD_mean[idx] = JSD_mean
            
            idx += 1

        # Mean over all test files
        mean_l1_final = np.mean(array_l1_final)
        std_l1_final = np.std(array_l1_final)
        mean_l1_mean = np.mean(array_l1_mean)
        std_l1_mean = np.std(array_l1_mean)
        mean_JSD_final = np.mean(array_JSD_final)
        std_JSD_final = np.std(array_JSD_final)
        mean_JSD_mean = np.mean(array_JSD_mean)
        std_JSD_mean = np.std(array_JSD_mean)

        with open(outfile, 'a') as f:
            if write_header:
                f.write('theta,shift,alpha_scale,mean_l1_final,std_l1_final,mean_l1_mean,std_l1_mean,mean_JSD_final,std_JSD_final,mean_JSD_mean,std_JSD_mean\n')
            f.write("%f,%f,%f,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e,%.3e\n" % (theta, shift, alpha_scale, mean_l1_final, std_l1_final, mean_l1_mean, std_l1_mean, mean_JSD_final, std_JSD_final, mean_JSD_mean, std_JSD_mean))

        return mean_l1_final, mean_l1_mean, mean_JSD_final, mean_JSD_mean


    def gridsearch(self, theta_range, shift_range, alpha_range, indir, outfile):
        """
        Arguments:
        theta_range - array
        shift_range - array
        alpha_range - array
        """
        list_tuples = [[100,0,0,0],[100,0,0,0],[100,0,0,0],[100,0,0,0]]
        for theta in theta_range:
            for shift in shift_range:
                for alpha_scale in alpha_range:
                    print("Theta %f, shift %f, alpha %d" % (theta, shift, alpha_scale))
                    result = self.evaluate(theta, shift, alpha_scale, indir=indir, outfile=outfile, write_header=0)
                    for idx in range(4):
                        if result[idx] <= list_tuples[idx][0]:
                            list_tuples[idx] = [result[idx], theta, shift, alpha_scale]
        print(list_tuples)


    def visualize(self, theta=8.86349, d=21, topic=0, dir_train='train_normalized', train_start=1, train_end=26, dir_test='test_normalized', test_start=27, test_end=37, save_plot=0, outfile='plots/mfg_topic0_theta8p86_s0p5_alpha1e4_m5d9.pdf'):
        """
        Run MFG policy forward using initial distributions across both training and test set,
        and plot trajectory of topic against all measurement data.
        """
        self.theta = theta
        self.d = d
        
        # Read train and test data
        df_train, df_test = self.var.read_data(dir_train, train_start, train_end, dir_test, test_start, test_end)

        # Generate trajectory from train data using policy
        print("Generating trajectory from train data")
        list_df = []
        idx = 0
        for num_day in range(train_start, train_end+1):
            # Read initial distribution pi0
            pi0 = np.array(df_train.iloc[(num_day-1)*16])

            # Generate entire trajectory using policy
            mat_trajectory = self.generate_trajectory(pi0, total_hours=16)
            df = pd.DataFrame(mat_trajectory)
            df.index = np.arange(idx, idx+16)
            list_df.append(df)
            idx += 16

        self.df_train_generated = pd.concat(list_df)
        self.df_train_generated.index = pd.to_datetime(self.df_train_generated.index, unit="D")

        # Generate trajectory using policy on test data
        print("Generating trajectory from test data")
        list_df = []
        for num_day in range(test_start, test_end+1):
            # Read initial distribution
            pi0 = np.array(df_test.iloc[(num_day-test_start)*16])
            # Generate entire trajectory using policy
            mat_trajectory = self.generate_trajectory(pi0, total_hours=16)
            df = pd.DataFrame(mat_trajectory)
            df.index = np.arange(idx, idx+16) # use same idx that was incremented above
            list_df.append(df)
            idx += 16
        self.df_test_generated = pd.concat(list_df)
        self.df_test_generated.index = pd.to_datetime(self.df_test_generated.index, unit="D")

        num_train = len(self.df_train_generated.index)
        array_x_train = np.arange(num_train)
        array_x_test = np.arange(num_train, num_train+len(self.df_test_generated.index))

        fig = plt.figure()
        plt.plot(array_x_train, df_train[topic], color='r', linestyle='-', label='train data')
        plt.plot(array_x_train, self.df_train_generated[topic], color='b', linestyle='--', label='MFG (train)')
        plt.plot(array_x_test, df_test[topic], color='k', linestyle='-', label='test data')
        plt.plot(array_x_test, self.df_test_generated[topic], color='g', linestyle='--', label='MFG (test)')        
        plt.ylabel('Topic %d popularity' % topic)
        plt.xlabel('Time steps (hrs)')
        plt.legend(loc='best')
        plt.title("Topic %d empirical and generated data" % topic)
        if save_plot == 1:
            pp = PdfPages(outfile)
            pp.savefig(fig)
            pp.close()
        else:
            plt.show()


    def read_rnn(self, path_to_file='rnn_normalized/trajectories_rnn.txt'):
        df = pd.read_csv(path_to_file, sep=' ', header=None, names=range(self.d), usecols=range(self.d), dtype=np.float32)
        df.index = pd.to_datetime(np.arange(0,160), unit="D")
        self.df_rnn = df

        
    def visualize_test(self, lag=13, theta=9.99, d=21, topic=0, dir_train='train_normalized_round2', train_start=1, train_end=18, dir_test='test_normalized_round2', test_start=19, test_end=24, mfg_and_rnn=0, log_scale=0,  save_plot=0, outfile='traj_mfg_var_0_8p06_0p16_12e3_13_m10d18.pdf'):
        """
        Produce plot of trajectory of raw test data, 
        MFG generated data, and time series prediction (from var.py)
        """
        self.theta = theta
        self.d = d
        
        # Read train and test data
        df_train, df_test = self.var.read_data(dir_train, train_start, train_end, dir_test, test_start, test_end)

        # Generate trajectory from test data using MFG policy
        print("Generating MFG trajectory from test data")
        idx = 0
        list_df = []
        for num_day in range(test_start, test_end+1):
            # Read initial distribution
            pi0 = np.array(df_test.iloc[(num_day-test_start)*16])
            # Generate entire trajectory using policy
            mat_trajectory = self.generate_trajectory(pi0, total_hours=16)
            df = pd.DataFrame(mat_trajectory)
            df.index = np.arange(idx, idx+16)
            list_df.append(df)
            idx += 16
        self.df_test_generated = pd.concat(list_df)
        self.df_test_generated.index = pd.to_datetime(self.df_test_generated.index, unit="D")

        # Train VAR and get forecast
        print("Running VAR to get forecast")
        self.var.train(lag, self.var.df_train)
        df_future_var = self.var.forecast(num_prior=int(16*(train_end-train_start+1)), steps=int(16*(test_end-test_start+1)), topic=topic, plot=0, show_plot=0)

        # Get RNN predictions
        self.read_rnn()

        #array_x_test = np.arange(0, len(self.df_test_generated.index))
        array_x_test = np.arange(0, len(self.df_test_generated.index))/16.0

        #fig = plt.figure()
        fig, ax = plt.subplots()
        for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] + ax.get_xticklabels() + ax.get_yticklabels()):
            item.set_fontsize(14)
            
        if mfg_and_rnn == 0:
            plt.plot(array_x_test, df_test[topic], color='k', linestyle='-', label='test data')
            plt.plot(array_x_test, self.df_test_generated[topic], color='g', linestyle='--', label='MFG (test)')
            plt.plot(array_x_test, df_future_var[topic], color='b', linestyle=':', label="VAR (test)")
        else:
            plt.plot(array_x_test, df_test[topic], color='k', linestyle='-', label='test data')
            plt.plot(array_x_test, self.df_test_generated[topic], color='g', linestyle='--', label='MFG (test)')
            plt.plot(array_x_test, self.df_rnn[topic], color='m', linestyle='-.', label='RNN (test)')
            if log_scale:
                ax.set_yscale('log')
        plt.ylabel('Topic %d popularity' % topic)
        plt.xlabel('Day')
        plt.xticks(np.arange(0,(test_end-test_start+1)+1,1))
        plt.legend(loc='best', prop={'size':14})
        plt.title("Topic %d measurement and predictions" % topic)

        if save_plot == 1:
            pp = PdfPages('plots_irl/'+outfile)
            pp.savefig(fig)
            pp.close()
        else:
            plt.show()

        

if __name__ == "__main__":
    ac = actor_critic(theta=8.86349, shift=0.16, alpha_scale=12000, d=21)
    t_start = time.time()
    ac.train(num_episodes=100000, gamma=1, lr_critic=0.1, lr_actor=0.1, consecutive=100, write_file=1, write_all=0)
    t_end = time.time()
    print("Time elapsed", t_end - t_start)
