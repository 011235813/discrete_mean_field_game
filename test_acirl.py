import numpy as np
import tensorflow as tf

import random

import ac_irl
ac = ac_irl.AC_IRL()

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
