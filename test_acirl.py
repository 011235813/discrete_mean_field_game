import numpy as np
import tensorflow as tf
from scipy.special import gamma

import random

import ac_irl

class test:

    def __init__(self):
        self.ac = ac_irl.AC_IRL()
        

    def dirichlet(alpha, x):
        num = 1
        for thing in alpha:
            num *= gamma(thing)
        denom = gamma( np.sum(alpha) )
    
        beta = num / denom
        print("num ", num)
        print("denom ", denom)
        print("beta ", beta)
    
        prod = 1
        for idx in range(len(alpha)):
            prod *= x[idx] ** (alpha[idx] - 1)
    
        return prod/beta
    
    
    def test_generate_trajectories(n):
        list_generated = ac.generate_trajectories(n)
        traj1 = list_generated[0]
        print("Length of trajectory: ", len(traj1))
        print("State 1")
        state1 = traj1[0][0]
        print(state1)
        print("Action 1")
        action1 = traj1[0][1]
        print(action1)
        print("State 2: ")
        state2 = traj1[1][0]
        print(state2)
        print("Calculated state 2")
        print(action1.T.dot(state1))
          
    
    def test_calc_pdf_action(theta):
        list_generated = ac.generate_trajectories(2)
        traj1 = list_generated[0]
        state1 = traj1[0][0]
        action1 = traj1[0][1]
        prob = ac.calc_pdf_action(theta, action1, state1)
        print(state1)
        print(prob)
    
    
    def test_network():
        idx_row = 0
        # select an initial start state
        pi = ac.mat_pi0[idx_row, :]
        print("Start state")
        print(pi)
        P = ac.sample_action(pi)
        print("Action matrix first row")
        print(P[0])
        reward = ac.sess.run( ac.reward_gen, feed_dict={ac.gen_states:[pi], ac.gen_actions:[P]} )
        print("Reward ", reward)
    
    
    def test_loss():
        ac.list_generated = ac.generate_trajectories(5 * ac.num_policies)
        list_generated = ac.generate_trajectories(5)
        ac.list_generated = ac.list_generated + list_generated
        # kick out trajectories generated from the earliest policy
        ac.list_generated = ac.list_generated[5: ]
    
        demo_sampled = random.sample(ac.list_demonstrations, ac.num_demo_samples)
        demo_states = [pair[0] for traj in demo_sampled for pair in traj]
        demo_actions = [pair[1] for traj in demo_sampled for pair in traj]
    
        gen_sampled = random.sample(ac.list_generated, ac.num_gen_samples)    
        gen_states = [pair[0] for traj in gen_sampled for pair in traj]
        gen_actions = [pair[1] for traj in gen_sampled for pair in traj]
    
        gen_states = gen_states + demo_states
        gen_actions = gen_actions + demo_actions
        
        feed_dict = {ac.demo_states:demo_states, ac.demo_actions:demo_actions, ac.gen_states:gen_states, ac.gen_actions:gen_actions}
    
        r_demo, r_gen = ac.sess.run( [ac.reward_demo, ac.reward_gen], feed_dict=feed_dict )
        print("r_demo")
        print(r_demo)
        print("r_gen")
        print(r_gen)
        print(len(r_demo), len(r_gen))
    
        actions_duplicated, tensor_alpha, pdf, reduce1, reduce2, reduce3, vec_z = ac.sess.run( [ac.actions_duplicated, ac.tensor_alpha, ac.pdf, ac.reduce1, ac.reduce2, ac.reduce3, ac.vec_z], feed_dict=feed_dict)
        print("tensor_alpha")
        print( np.isnan(tensor_alpha).any() )
        print("pdf")
        print( np.isnan(pdf).any() )
        print("location of NaN in pdf")
        idx_nan = np.argwhere(np.isnan(pdf))
        print(idx_nan)
        if idx_nan.size != 0:
            print("pdf is NaN at location")
            row = idx_nan[0]
            print(row)
            print("tensor_alpha at that location is")
            print( tensor_alpha[row[0],row[1],row[2],row[3]] )
            print("actions_duplicated at that location is")
            print( actions_duplicated[row[0],row[1],row[2],row[3]] )
        # print("min in actions_duplicated ", np.min(actions_duplicated))
        # print("min in tensor_alpha ", np.min(tensor_alpha))
        print("reduce1")
        print( np.isnan(reduce1).any() )
        print("reduce2")
        print( np.isnan(reduce2).any() )
        print("reduce3")
        print( np.isnan(reduce3).any() )
        print("vec_z")
        print( np.isnan(vec_z).any() )    
    
    
    def test_calc_z(self, c):
        self.ac.list_generated = self.ac.generate_trajectories(5 * self.ac.num_policies)
        list_generated = self.ac.generate_trajectories(5)
        self.ac.list_generated = self.ac.list_generated + list_generated
        # kick out trajectories generated from the earliest policy
        self.ac.list_generated = self.ac.list_generated[5: ]
    
        demo_sampled = random.sample(self.ac.list_demonstrations, self.ac.num_demo_samples)
        demo_states = [pair[0] for traj in demo_sampled for pair in traj]
        demo_actions = [pair[1] for traj in demo_sampled for pair in traj]
    
        gen_sampled = random.sample(self.ac.list_generated, self.ac.num_gen_samples)    
        gen_states = [pair[0] for traj in gen_sampled for pair in traj]
        gen_actions = [pair[1] for traj in gen_sampled for pair in traj]
    
        # gen_states = gen_states + demo_states
        # gen_actions = gen_actions + demo_actions
        
        # feed_dict = {self.ac.demo_states:demo_states, self.ac.demo_actions:demo_actions, self.ac.gen_states:gen_states, self.ac.gen_actions:gen_actions, self.ac.c:c}
        self.feed_dict = {self.ac.demo_states:demo_states, self.ac.demo_actions:demo_actions, self.ac.gen_states:gen_states, self.ac.gen_actions:gen_actions, self.ac.c:c}    
        
        sum_demo_rewards = - 1.0/self.ac.num_demo_samples * tf.reduce_sum(self.ac.reward_demo)
        sum_demo_rewards_val = self.ac.sess.run(sum_demo_rewards, feed_dict=self.feed_dict)
        print("sum_demo_rewards = ", sum_demo_rewards_val)
    
        gen_rewards_reshaped = tf.reshape( self.ac.reward_gen, [self.ac.num_sampled_trajectories, 15] )
        # sum over time to get vector of r(traj_sample) for each traj_sample
        gen_rewards_per_traj = tf.reduce_sum( gen_rewards_reshaped, axis=1 )
        # exponentiate to get exp(r(traj_sample))
        gen_rewards_exp = tf.exp(gen_rewards_per_traj) # [num_sampled_trajectories]
        gen_rewards_exp_val = self.ac.sess.run(gen_rewards_exp, feed_dict=self.feed_dict)
        print("gen_rewards_exp = ")
        print(gen_rewards_exp_val)
    
        self.actions_duplicated_val = self.ac.sess.run(self.ac.actions_duplicated, feed_dict=self.feed_dict)
        self.tensor_alpha_val = self.ac.sess.run(self.ac.tensor_alpha, feed_dict=self.feed_dict)
        self.tensor_alpha_lowerbound_val = self.ac.sess.run(self.ac.tensor_alpha_lowerbound, feed_dict=self.feed_dict)

        self.pdf_val = self.ac.sess.run(self.ac.pdf, feed_dict=self.feed_dict)
        print("pdf max val = ", np.max(self.pdf_val))
        print("pdf min val = ", np.min(self.pdf_val))
        pdf_max_along_rows = np.max(self.pdf_val, axis=3)
        pdf_min_along_rows = np.min(self.pdf_val, axis=3)
        pdf_diff = pdf_max_along_rows - pdf_min_along_rows
        pdf_max_diff = np.max(pdf_diff)
        print("pdf max over everything of difference between max and min val for a single P matrix ", pdf_max_diff)
    
        pdf_normalized_val = self.ac.sess.run(self.ac.pdf_normalized, feed_dict=self.feed_dict)
        print("pdf_normalized[0,0,0]")
        print(pdf_normalized_val[0,0,0])
    
        pdf_normalized_val_max = np.max(pdf_normalized_val)
        print("max val in pdf_normalized = ", pdf_normalized_val_max)
    
        self.reduce1_val = self.ac.sess.run(self.ac.reduce1, feed_dict=self.feed_dict)
        print("reduce1[0,0]")
        print(self.reduce1_val[0,0])
    
        self.reduce2_val = self.ac.sess.run(self.ac.reduce2, feed_dict=self.feed_dict)
        print("reduce2[0]")
        print(self.reduce2_val[0])
    
        self.reduce3_val = self.ac.sess.run(self.ac.reduce3, feed_dict=self.feed_dict)
        print("reduce3")
        print(self.reduce3_val)
    
        self.vec_z_val = self.ac.sess.run(self.ac.vec_z, feed_dict=self.feed_dict)
        print("vec_z")
        print(self.vec_z_val)
