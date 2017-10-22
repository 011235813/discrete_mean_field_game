import ac_irl
import tensorflow as tf


with open('results/reward_gridsearch_10_22.csv', 'w') as f:
    f.write('reg,n_fc3,n_fc4,reward_demo_avg_train,reward_demo_avg_test,reward_gen_avg,theta\n')

list_reg = ['dropout', 'l1l2', 'dropout_l1l2']
list_nfc3 = range(4,10,2)
list_nfc4 = range(4,10,2)

for reg in list_reg:

    for n_fc3 in list_nfc3:

        for n_fc4 in list_nfc4:

            print("---------- reg = %s | n_fc3 = %d | n_fc4 = %d ----------" % (reg, n_fc3, n_fc4))

            # Train
            tf.reset_default_graph()
            ac = ac_irl.AC_IRL(theta=6.5, reg=reg, n_fc3=n_fc3, n_fc4=n_fc4)
            final_theta = ac.outerloop()
            
            # Evaluate reward
            # tf.reset_default_graph()
            # ac = ac_irl.AC_IRL(theta=final_theta, reg=reg, n_fc3=n_fc3, n_fc4=n_fc4, saved_network='model_%s_%d_%d.ckpt' % (reg, n_fc3, n_fc4))
            reward_demo_avg_train, reward_demo_avg_test, reward_gen_avg = ac.test_reward_network()
            
            with open('results/reward_gridsearch_10_22.csv', 'a') as f:
                f.write('%s,%d,%d,%f,%f,%f,%f\n' % (reg, n_fc3, n_fc4, reward_demo_avg_train, reward_demo_avg_test, reward_gen_avg, final_theta))
