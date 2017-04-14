import numpy as np
import mfg_ac2
import time

ac = mfg_ac2.actor_critic(theta=10, shift=0.4, d=4)

def test_action():

    pi = np.array([0.7, 0.09, 0.01, 0.2])
    
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

def time_action(n):
    ac.init_pi0(r'C:\Users\Jiachen\Documents\Projects\Python\RL\MFG\data_train_reordered')
    ac.num_start_samples = ac.mat_pi0.shape[0]
    idx_row = ac.num_start_samples - 1
    pi = ac.mat_pi0[idx_row, :]
    t_start = time.time()
    for j in range(n):
        P = ac.sample_action(pi)
    t_end = time.time()
    print("Time taken for %d runs of sample_action" % n, (t_end - t_start))
    

def test_cost():
    ac = mfg_ac2.actor_critic()

    P = np.array([[1,3,3],[4,5,6],[7,8,9]])
    print(P)
    pi = np.array([0.1,0.2,0.7])
    print(pi)

    reward = ac.calc_cost(P, pi, 3)

    print("Reward", reward)

def time_cost(n):
    ac = mfg_ac2.actor_critic()
    # P = np.array([[1,3,3,4],[4,5,6,6],[7,8,9,10],[1,2,3,4]])
    P = np.ones([47,47])
    # pi = np.array([0.1,0.2,0.6,0.1])
    pi = np.zeros(47)
    pi[0] = 1
    t_start = time.time()
    for j in range(n):
        reward = ac.calc_cost(P, pi, 47)
    t_end = time.time()
    print("Time taken for %d runs of calc_cost" % n, (t_end - t_start))
    print("Reward", reward)

    
def test_value():
    d = 3
    ac = mfg_ac2.actor_critic(d=d)
    num_features = int((d+1)*d/2 + d + 1)
    ac.w = np.ones(num_features)
    print("weights")
    print(ac.w)

    pi = np.array([0.1,0.2,0.7])
    print("pi")
    print(pi)

    value = ac.calc_value(pi)
    # vec_features = ac.calc_features(pi)
    # value = vec_features.dot(ac.w)
    print("value", value)

def time_value(n):
    d = 47
    ac = mfg_ac2.actor_critic(d=d)
    num_features = int((d+1)*d/2 + d + 1)
    ac.w = np.ones(num_features)
    pi = np.random.rand(d)

    t_start = time.time()
    for j in range(n):
        value = ac.calc_value(pi)
    t_end = time.time()
    print("Time taken for %d runs of calc_value" % n, (t_end - t_start))
    print("Value", value)


def test_gradient_first():
    ac.theta = 10
    ac.d = 4
    pi = np.array([0.7,0.09,0.01,0.2])

    P = ac.sample_action(pi)
    print("P")
    print(P)
    
    gradient1 = ac.calc_gradient_basic(P, pi)
    print("gradient basic", gradient1)

    gradient2 = ac.calc_gradient(P, pi)
    print("gradient", gradient2)
    

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

    gradient_2 = ac.calc_gradient_vectorized(P, pi)
    print("Gradient computed by vectorized code")
    print(gradient_2)

def test_gradient_small():
    ac.mat_alpha = np.array([[1,2],[3,4]])
    ac.tensor_phi = np.array([[[1,1],[2,2]],[[3,3],[4,4]]])
    ac.theta = np.array([[5],[6]])
    P = np.array([[0.2,0.8],[0.4,0.6]])
    pi = np.array([0.1,0.9])
    ac.dim_theta = 2
    ac.d = 2
    
    gradient = ac.calc_gradient(P, pi)
    print("Gradient vector")
    print(gradient)

    gradient_2 = ac.calc_gradient_vectorized(P, pi)
    print("Gradient computed by vectorized code")
    print(gradient_2)    

    gradient_3 = ac.calc_gradient_basic(P, pi)
    print("Gradient computed by basic code")
    print(gradient_3)        

def time_gradient(n):
    ac.init_pi0(r'C:\Users\Jiachen\Documents\Projects\Python\RL\MFG\data_train_reordered')
    ac.num_start_samples = ac.mat_pi0.shape[0]
    idx_row = np.random.randint(ac.num_start_samples)
    pi = ac.mat_pi0[idx_row, :]
    P = ac.sample_action(pi)
    
    t_start = time.time()
    for j in range(n):
        gradient = ac.calc_gradient(P, pi)
    t_end = time.time()
    print("Time taken by loop version", t_end-t_start)
    print("Gradient computed by loop version")
    print(gradient)

    t_start = time.time()
    for j in range(n):
        gradient = ac.calc_gradient_vectorized(P, pi)
    t_end = time.time()
    print("Time taken by vector version", t_end-t_start)
    print("Gradient computed by vectorized version")
    print(gradient)    

    
def test_forward(n, theta, shift, alpha_scale):
    ac.theta = theta
    ac.shift = shift
    ac.d = 47
    ac.alpha_scale = alpha_scale
    pi = np.zeros(47)
    pi[0] = 0.9
    pi[1:] = np.ones(ac.d-1) * 0.1 / (ac.d-1)

    print("pi")
    print(pi)

    for i in range(n):
        print("Iteration %d" % i)
        
        P = ac.sample_action(pi)
        print("P")
        print(P)

        pi = np.transpose(P).dot(pi)
        print("pi")
        print(pi)
