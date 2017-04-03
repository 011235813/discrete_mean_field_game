import numpy as np
import mfg_ac

ac = mfg_ac.actor_critic()

def test_action():

    ac.init_pi0(r'C:\Users\Jiachen\Documents\Projects\Python\RL\MFG\data_train_reordered')
    
    print("mat_pi0")
    print(ac.mat_pi0)
    
    ac.num_start_samples = ac.mat_pi0.shape[0]
    
    idx_row = np.random.randint(ac.num_start_samples)
    
    print("idx_row", idx_row)
    
    pi = ac.mat_pi0[idx_row, :]
    
    print("pi")
    print(pi)
    print("Sum of pi", np.sum(pi))
    
    P = ac.sample_action(pi)
    
    print("P matrix")
    print(P)
    print("Row 1 sum", np.sum(P[1, :]))
    print("Col 3 sum", np.sum(P[:, 3]))
    
    print("New pi")
    pi_next = np.transpose(P).dot(pi)
    print(pi_next)
    print("Row sum", np.sum(pi_next))


def test_cost():
    ac = mfg_ac.actor_critic()

    P = np.array([[1,3,3],[4,5,6],[7,8,9]])
    print(P)
    pi = np.array([0.1,0.2,0.7])
    print(pi)

    reward = ac.calc_cost(P, pi, 3)

    print("Reward", reward)

def test_value():
    ac = mfg_ac.actor_critic(d=3)

    print("weights")
    print(ac.w)

    pi = np.array([0.1,0.2,0.7])
    print("pi")
    print(pi)

    # value = ac.calc_value(pi)
    vec_features = ac.calc_features(pi)
    value = vec_features.dot(ac.w)
    print("value", value)


def test_gradient():

    ac.init_pi0(r'C:\Users\Jiachen\Documents\Projects\Python\RL\MFG\data_train_reordered')
    
    print("mat_pi0")
    print(ac.mat_pi0)
    
    ac.num_start_samples = ac.mat_pi0.shape[0]
    
    idx_row = np.random.randint(ac.num_start_samples)
    
    print("idx_row", idx_row)
    
    pi = ac.mat_pi0[idx_row, :]
    
    print("pi")
    print(pi)
    print("Sum of pi", np.sum(pi))
    
    P = ac.sample_action(pi)
    
    print("P matrix")
    print(P)
    print("Row 1 sum", np.sum(P[1, :]))
    print("Col 3 sum", np.sum(P[:, 3]))
    
    gradient = ac.calc_gradient(P, pi)
    print("Gradient vector")
    print(gradient)
