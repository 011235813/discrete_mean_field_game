"""
Inverse reinforcement learning with policy improvement in the loop
Uses the MaxEnt IRL guided cost learning algorithm in Finn et al. 2016
"""

import numpy as np
from numpy.linalg import norm
import tensorflow as tf

from scipy import special
from scipy.stats import entropy
# from scipy.stats import dirichlet
import functools

import platform
if (platform.system() == "Windows"):
    import pandas as pd
    import matplotlib.pylab as plt
    from matplotlib.backends.backend_pdf import PdfPages    
    import var

import os
import itertools
import time
import random

import networks

class AC_IRL:

    def __init__(self, theta=8.86349, shift=0.16, alpha_scale=12000, d=20, lr_reward=1e-4, num_policies=10):

        # initialize theta
        self.theta = theta
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

        # initialize collection of start states
        self.init_pi0(path_to_dir=os.getcwd()+'/train_normalized_round2')
        self.num_start_samples = self.mat_pi0.shape[0] # number of rows

        # Will become list of list of tuples of the form (state, action)
        self.list_demonstrations = []
        self.read_demonstrations(state_dir='./train_normalized_round2', action_dir='./actions')

        # This is D_samp in the IRL algorithm. Will be populated while running outerloop()
        self.list_generated = []

        # Create neural net representation of reward function
        self.create_network()

        # number of demonstration trajectories to sample each time for reward learning
        self.num_demo_samples = 5
        # number of generated trajectories to sample each time for reward learning
        self.num_gen_samples = 5
        # number of trajectories to use for part 2 of loss function
        self.num_sampled_trajectories = self.num_demo_samples + self.num_gen_samples
        # list of policies parameterized by theta, beginning with the initialized theta
        self.list_policies = [theta] * self.num_policies

        # vector of importance sampling weights in loss function
        # self.vec_z = tf.Variable(np.zeros(self.num_sampled_trajectories), dtype=tf.float32)

        # Define gradient descent steps for reward learning
        self.create_training_method()

        self.sess = tf.Session()
        init = tf.global_variables_initializer()
        self.sess.run(init)
        
    # ------------------- File processing functions ------------------ #

    # ------------------- End file processing functions ------------------ #


    # ------------------- New and overridden functions ---------------- #

    def read_demonstrations(self, state_dir, action_dir):
        """
        Reads measured trajectories to produce list of trajectories,
        where each trajectory is a list of (state, action) pairs,
        where each state is a vector and action is a transition matrix
        
        state_dir - name of folder containing measured states for each hour of each day
        action_dir - name of folder containing measured actions for each hour of each day
        """
        print("Inside read_demonstrations")
        num_file_action = len(os.listdir(action_dir))
        num_file_state = len(os.listdir(state_dir))
        if num_file_action != num_file_state:
            print("Weird")
            
        for idx_day in range(1, 1+num_file_action):
            trajectory = []
            file_state = "trend_distribution_day%d.csv" % idx_day
            file_action = "action_day%d.txt" % idx_day
            # read states for this day
            states = pd.read_table(state_dir+'/'+file_state, delimiter=' ', header=None)
            # convert to np matrix
            states = states.as_matrix() # 16 x 20
            # read actions for this day, automatically skips over blank lines
            actions = pd.read_table(action_dir+'/'+file_action, delimiter=' ', header=None)
            actions = actions.as_matrix() # (15*20) * 20
            # group into (state, action) pairs
            for hour in range(0,15):
                state = states[hour,:]
                action = actions[hour*self.d:(hour+1)*self.d, :]
                # append state-action pair
                trajectory.append( (state, action) )
            # append to list of demonstrations
            self.list_demonstrations.append( trajectory )


    def create_network(self):
        """
        Creates neural net representation of reward function
        """
        print("Inside create_network")
        # placeholder for actions in demonstration batch, shape [N, num_actions, d,d]
        # where N = number of trajectories * num actions along trajectory (should be 15)
        self.demo_actions = tf.placeholder(dtype=tf.float32, shape=[None,self.d,self.d])
        # placeholder for states in demonstration batch
        # where N = number of trajectories * num states along trajectory (should be 15)
        self.demo_states = tf.placeholder(dtype=tf.float32, shape=[None,self.d])
        # placeholder for actions in generated batch
        self.gen_actions = tf.placeholder(dtype=tf.float32, shape=[None,self.d,self.d])
        # placeholder for states in generated batch
        self.gen_states = tf.placeholder(dtype=tf.float32, shape=[None,self.d])
        with tf.variable_scope("reward") as scope:
            # rewards for state-action pairs in demonstration batch
            # expected dimension [N, 1] where N = total number of transitions in batch
            self.reward_demo = networks.r_net(self.demo_states, self.demo_actions, f1=2, k1=5, f2=4, k2=3)
            scope.reuse_variables()
            # rewards for state-action pairs in generated batch
            # expected dimension [N, 1] where N = total number of transitions in batch
            self.reward_gen = networks.r_net(self.gen_states, self.gen_actions, f1=2, k1=5, f2=4, k2=3)


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
        # self.gen_actions is [num_sampled_trajectories*15, 20, 20]
        # reshape to [num_sampled_trajectories, 15, 20, 20]
        gen_actions_reshaped = tf.reshape(self.gen_actions, [self.num_sampled_trajectories,15,self.d,self.d])
        # self.gen_states is [num_sampled_trajectories*15, 20]
        # reshape to [num_sampled_trajectories, 15, 20]
        gen_states_reshaped = tf.reshape(self.gen_states, [self.num_sampled_trajectories,15,self.d])

        list_ops = []
        for j in range(self.num_sampled_trajectories):
            traj_actions = gen_actions_reshaped[j] # 15 x 20 x 20
            traj_states = gen_states_reshaped[j] # 15 x 20
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
        # self.gen_actions is [num_trajectories x 15, 20, 20]
        # Reshape actions to [num_trajectories, 1, time, 20, 20] with an extra dimension
        actions_reshaped = tf.reshape(self.gen_actions, [self.num_sampled_trajectories, 1, 15, self.d, self.d])
        # duplicate along the extra dimension <self.num_policies> times
        # [<num_trajectories>, <num_policies>, 15, 20, 20]
        self.actions_duplicated = tf.tile( actions_reshaped, [1,self.num_policies,1,1,1] )
        # replace all instances of 0 with 1e-6 to avoid raising 0 to negative power when
        # calculating pdf
        self.actions_nozeros = tf.where( tf.equal(self.actions_duplicated,0), tf.ones_like(self.actions_duplicated)*1e-6, self.actions_duplicated)

        # Use self.gen_states to create alpha tensor
        # self.gen_states is [num_trajectories x 15 , 20]
        # [<num_trajectories>, 15, 20, 20], all rows over 3rd dimension are same
        # Compare to mat1 in matrix case
        states_duplicated_rows = tf.tile( tf.reshape(self.gen_states, [self.num_sampled_trajectories, 15, 1, self.d]), [1,1,self.d,1] )
        # [<num_trajectories>, 15, 20, 20], all columns over 4th dimension are same
        # Compare to mat2 in matrix case
        states_duplicated_columns = tf.tile( tf.reshape(self.gen_states, [self.num_sampled_trajectories, 15, self.d, 1]), [1,1,1,self.d] )
        diff = states_duplicated_rows - states_duplicated_columns
        # Duplicate along one extra dimension <self.num_policies> times
        # [<num_trajectories>, <num_policies>, 15, 20, 20]
        diff_duplicated = tf.tile( tf.reshape(diff, [self.num_sampled_trajectories,1,15,self.d,self.d]), [1,self.num_policies,1,1,1] )
        # Weight by each policy's theta
        theta_duplicated = tf.reshape( tf.cast(self.list_policies, tf.float32), [1,self.num_policies, 1,1,1] )
        self.tensor_alpha = tf.log( 1 + tf.exp( tf.multiply((diff_duplicated - self.shift), theta_duplicated) ) ) # [<num_trajectories>, <num_policies>, 15, 20, 20]
        # Compute dirichlet for everything
        dist = tf.distributions.Dirichlet(self.tensor_alpha)
        self.pdf = dist.prob(self.actions_nozeros) # [<num_trajectories>, <num_policies>, 15, 20]

        # Now reduce everything
        # product over topic dimension to get q_k(P^t) from q_k(P^t_1)...q_k(P^t_d)
        self.reduce1 = tf.reduce_prod(self.pdf, axis=3) # [<num_trajectories>, <num_policies>, 15]
        # product over time to get q_k(tau_j) from q_k(P^1)...q_k(P^15)
        # NOTE: start state probability Pr(s_1) will be multiplied in below
        self.reduce2 = tf.reduce_prod(self.reduce1, axis=2) # [<num_trajectories>, <num_policies>]
        # sum over policy dimension to get sum_k q_k(tau_j)
        self.reduce3 = tf.reduce_sum(self.reduce2, axis=1) # [<num_trajectories>]
        # z_j = k / (sum_k q_k(tau_j)), along with start state probability
        self.vec_z = self.num_policies / ( self.num_start_samples * self.reduce3 )
        

    def create_training_method(self):
        """
        Defines loss, optimizer, and creates training operation.
        """
        print("Inside create_training_method")
        # first term = - 1/N sum_{traj_demo} r(traj_demo)
        # where N = number of sampled demo trajectories
        # just sum up r(s,a) over all (s_t,a_t) in each trajectory over all demo trajectories
        sum_demo_rewards = - 1.0/self.num_demo_samples * tf.reduce_sum(self.reward_demo)

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
        self.calc_z()
        second_term = tf.log( 1.0/self.num_sampled_trajectories * tf.reduce_sum( self.vec_z * gen_rewards_exp) )

        # compute loss = negative log likelihood
        self.loss = sum_demo_rewards + second_term
        optimizer = tf.train.AdamOptimizer(learning_rate=self.lr_reward)
        self.r_train_op = optimizer.minimize(self.loss)


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
    

    def train(self, num_episodes=4000, gamma=1, constant=False, lr_critic=0.1, lr_actor=0.001, consecutive=100, file_theta='results/theta.csv', file_pi='results/pi.csv', file_reward='results/reward.csv', write_file=0, write_all=0):
        """
        Main actor-critic training procedure that improves theta and w

        Input:
        num_episodes - each episode is 16 steps (9am to 12midnight)
        gamma - temporal discount
        constant - if True, then does not decrease learning rates
        lr_critic - learning rate for value function parameter update
        lr_actor - learning rate for policy parameter update
        consecutive - number of consecutive episodes between report of average reward
        """
        print("Running forward solver")
        list_reward = []
        for episode in range(1, num_episodes+1):
            print("forward episode ", episode)
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
                print("Forward solver step %d before sample_action" % num_steps)
                P = self.sample_action(pi)
                print("Forward solver step %d pi" % num_steps)
                print(pi)
                print("Forward solver step %d action matrix row 0" % num_steps)
                print(P[0])

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
                print("Forward solver step %d reward" % num_steps)
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
                print("Forward solver step %d before theta update" % num_steps)
                print(delta)
                print(gradient)
                if constant:
                    self.theta = self.theta + lr_actor * delta * gradient
                else:
                    self.theta = self.theta + (lr_actor/((episode+1)*np.log(np.log(episode+20)))) * delta * gradient #here
                print("Forward solver step %d after theta update" % num_steps)

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

        # record this policy
        self.list_policies = (self.list_policies + [self.theta])[1:]


    def generate_trajectories(self, n):
        """
        Use the current policy self.theta to generate trajectories
        n - number of trajectories to generate

        Return: list of generated trajectories
        """
        print("Inside generate_trajectories")
        print("Inside generate_trajectories, theta ", self.theta)
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


    def update_reward(self):
        """
        Improvement of reward function via gradient descent

        """
        print("Updating reward")
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
        gen_states = gen_states + demo_states
        gen_actions = gen_actions + demo_actions

        feed_dict = {self.demo_states:demo_states, self.demo_actions:demo_actions, self.gen_states:gen_states, self.gen_actions:gen_actions}

        # Execute gradient descent
        print("Running r_train_op")
        _ = self.sess.run(self.r_train_op, feed_dict=feed_dict)


    def outerloop(self, num_iterations=1000, num_gen_from_policy=5, num_forward_episodes=100, gamma=1, constant=False, lr_critic=0.1, lr_actor=0.001):
        """
        Outer-most loop that calls functions to update reward function
        and solve the forward problem

        num_iterations - number of reward function updates
        num_generated - number of trajectories to generate using policy at each step
        num_forward_episodes - number of episodes to run when solving forward problem
        gamma - temporal discount
        constant - if True, then does not decrease learning rates
        lr_critic - learning rate for value function parameter update
        lr_actor - learning rate for policy parameter update
        """

        # At the beginning, generate trajectories from initial policies, all of which
        # are the same. This is meant to populate the data for use in reward learning,
        # before accumulating <num_policies> different policies from forward passes
        self.list_generated = self.generate_trajectories(num_gen_from_policy * self.num_policies)

        for it in range(num_iterations):
            print("Iteration ", it)
            # Generate samples D_traj from current policy
            list_generated = self.generate_trajectories(num_gen_from_policy)

            # D_samp <- D_samp union D_traj
            self.list_generated = self.list_generated + list_generated
            # kick out trajectories generated from the earliest policy
            self.list_generated = self.list_generated[num_gen_from_policy: ]

            print("Before update_reward, theta ", self.theta)
            # Update reward function
            self.update_reward()
            print("After update_reward, theta ", self.theta)

            # Solve forward problem
            self.train(num_forward_episodes, gamma, constant, lr_critic, lr_actor, consecutive=10, file_theta='results/theta.csv', file_pi='results/pi.csv', file_reward='results/reward.csv', write_file=1, write_all=0)
            print("\n")


# ---------------- End training code ---------------- #

# ---------------- Evaluation code ------------------ #

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


    def evaluate(self, theta=8.86349, shift=0.5, alpha_scale=1e4, d=21, episode_length=16, indir='test_normalized', outfile='test_eval.csv', write_header=0):
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

        
    def visualize_test(self, theta=9.99, d=21, topic=0, dir_train='train_normalized2', train_start=1, train_end=35, dir_test='test_normalized2', test_start=36, test_end=45, mfg_and_rnn=0, log_scale=0,  save_plot=0, outfile='plots/mfg_var_0_9p99_0p02_2e4_22_m5d15.pdf'):
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
        self.var.train(22, self.var.df_train)
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
        plt.xticks(np.arange(0,10+1,1))
        plt.legend(loc='best', prop={'size':14})
        plt.title("Topic %d measurement and predictions" % topic)

        if save_plot == 1:
            pp = PdfPages(outfile)
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
