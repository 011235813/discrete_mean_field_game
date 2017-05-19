# This version is meant to test the correctness of RL versus MFG equations
# It uses reward function rather than cost function
# Reward function is different from the one in Gomes paper

import numpy as np
from numpy.linalg import norm

from scipy import special
from scipy.stats import entropy

import platform
if (platform.system() == "Windows"):
    import pandas as pd
    import matplotlib.pylab as plt
    import var

import os
import itertools
import time
import warnings

warnings.filterwarnings('error')

class actor_critic:

    def __init__(self, theta=10, shift=0, alpha_scale=100, d=21):

        # initialize theta
        self.theta = theta
        self.shift = shift
        self.alpha_scale = alpha_scale

        # initialize weight vector (column) for value function approximation
        self.w = self.init_w(d)

        # number of topics
        self.d = d

        # d x d x dim_theta tensor, computed within sample_action and used for
        # calculating gradient for theta update
        # self.tensor_phi = np.zeros([self.d, self.d, self.dim_theta])        #here

        # d x d matrix, computed within sample_action and used for sampling action P
        # and also for calculating gradient for theta update
        self.mat_alpha = np.zeros([self.d, self.d])

        self.mat_alpha_deriv = np.zeros([self.d, self.d])

        if (platform.system() == "Windows"):
            self.var = var.var(d=d)

# ------------------- File processing functions ------------------ #


    def reorder(self, list_rows):
        """
        Given a list of rows (each is a pi^n), order all rows by decreasing popularity
        based on the first row.
        """
        row1 = list_rows[0]
        # create mapping from index to value
        list_pairs = []
        for i in range(len(row1)):
            list_pairs.append( (i, row1[i]) )

        # sort by decreasing popularity
        list_pairs.sort(reverse=True, key=lambda x: x[1])

        # extract ordering
        order = []
        for pair in list_pairs:
            order.append( pair[0] )
        
        # apply ordering to all rows in list_rows
        for i in range(len(list_rows)):
            list_rows[i] = [ list_rows[i][j] for j in order ]

        return list_rows

    
    def reorder_files(self, indir='train', outdir='train_reordered'):
        """
        Process all files in given directory, creates new files
        """
        path_to_dir = os.getcwd() + '/' + indir
        path_to_outdir = os.getcwd() + '/' + outdir

        for filename in os.listdir(path_to_dir):
            path_to_file = path_to_dir + '/' + filename
            f = open(path_to_file, 'r')
            f.readline() # skip the header line of topics
            list_lines = f.readlines()
            f.close()
            # strip away newline, convert csv format to list of entries,
            # remove the last empty entry (due to extra comma)
            list_lines = list(map(lambda x: x.strip().split(',')[:-1], list_lines))
            # convert to int
            for i in range(len(list_lines)):
                list_lines[i] = list(map(int, list_lines[i]))
            # reorder
            list_rows = self.reorder(list_lines)
            # write to new file
            index_dot = filename.index('.')
            filename_new = filename[:index_dot] + '_reordered' + filename[index_dot:]
            f = open(path_to_outdir + '/' + filename_new, 'w')
            for row in list_rows:
                s = ','.join(map(str, row))
                s += '\n'
                f.write(s)
            f.close()


    def normalize(self, indir='train_reordered', outdir='train_normalized'):
        """
        Normalize all rows of data
        """
        path_to_dir = os.getcwd() + '/' + indir
        path_to_outdir = os.getcwd() + '/' + outdir
        for filename in os.listdir(path_to_dir):
            path_to_file = path_to_dir + '/' + filename
            with open(path_to_file, 'r') as f:
                matrix = np.loadtxt(f, delimiter=',')
            num_rows = matrix.shape[0]
            matrix = matrix / np.sum(matrix, axis=1).reshape(num_rows,1)
            path_to_outfile = path_to_outdir + '/' + filename
            with open(path_to_outfile, 'wb') as f:
                np.savetxt(f, matrix, fmt='%.3e')


    def get_max_nonzero(self, indir):
        """
        Scan through the training files in indir and find the maximum
        number of nonzero entries in the initial distribution
        """
        max_nnz = 0
        file_with_max = ''
        path_to_dir = os.getcwd() + '/' + indir
        for filename in os.listdir(path_to_dir):
            path_to_file = path_to_dir + '/' + filename
            with open(path_to_file, 'r') as f:
                matrix = np.loadtxt(f, delimiter=',')
            nnz = np.count_nonzero(matrix[0])
            if ( nnz > max_nnz ):
                max_nnz = nnz
                file_with_max = filename
        print("Max nnz:", max_nnz)
        print("File with max nnz:", file_with_max)


# ------------------- End file processing functions ------------------ #


# ------------------- Actor-critic training functions ---------------- #

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
        pi^0_1, ... , pi^0_d
        where d is a fixed constant across all files
        """
        list_pi0 = []
        # for filename in os.listdir(path_to_dir):
        num_files = len(os.listdir(path_to_dir))
        for num_day in range(1, 1+num_files):
            filename = "trend_distribution_day%d_reordered.csv" % num_day
            path_to_file = path_to_dir + '/' + filename
            f = open(path_to_file, 'r')
            list_lines = f.readlines()
            f.close()
            # Need to decide whether or not to include the null topic at index 0
            list_pi0.append( list(map(float, list_lines[0].strip().split(' ')))[0:self.d] )
            if verbose:
                print(filename)
            
        num_rows = len(list_pi0)
        num_cols = len(list_pi0[0])

        self.mat_pi0 = np.zeros([num_rows, num_cols])
        for i in range(len(list_pi0)):
            # total = np.sum(list_pi0[i])
            # self.mat_pi0[i] = list(map(lambda x: x/total, list_pi0[i]))
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
        # Construct all derivatives
        self.mat_alpha_deriv = np.zeros([self.d, self.d])
        # Create tensor phi(i,j,pi) for storing all phi matrices for later use
        # self.tensor_phi = np.zeros([self.d,self.d,self.dim_theta])

        # temp_{ij} = pi_j - pi_i
        # alpha^i_j = ln ( 1 + exp[ theta ( (pi_j - pi_i) - shift ) ] )
        mat1 = np.repeat(pi.reshape(1, self.d), self.d, 0)
        mat2 = np.repeat(pi.reshape(self.d, 1), self.d, 1)
        temp = mat1 - mat2
        self.mat_alpha = np.log( 1 + np.exp( self.theta * (temp - self.shift)))
        
        # Also create matrix of derivatives
        # d(alpha^i_j)/d(theta) = \frac{ pi_j - pi_i - shift } { 1 + exp( -theta*(pi_j - pi_i - shift) ) }
        numerator = temp - self.shift
        denominator = 1 + np.exp( (-self.theta) * numerator)
        self.mat_alpha_deriv = numerator / denominator

        # Sample matrix P from Dirichlet
        P = np.zeros([self.d, self.d])
        for i in range(self.d):
            # Get y^i_1, ... y^i_d
            # y = [np.random.gamma(shape=a, scale=1) for a in self.mat_alpha[i, :]]
            # Using the vector as input to shape reduces runtime by 5s
            y = np.random.gamma(shape=self.mat_alpha[i,:]*self.alpha_scale, scale=1)
            # replace zeros with dummy value
            y[y == 0] = 1e-20
            total = np.sum(y)
            # Store into i-th row of matrix P
            # P[i] = [y_j/total for y_j in y]
            try:
                P[i] = y / total
            except Warning:
                P[i] = y / total
                print(y, total)

        return P


    def calc_reward(self, P, pi, d):
        """
        Input:
        P - transition matrix
        pi - population distribution as row vector
        d - should be self.d always, except during testing

        Using r_i(pi, P_i) = - 1/2 ||P_i||^2, reward is
        R = - \sum_i pi_i 1/2 ||P_i||^2
        = - 1/2< pi , v >
        where v is vector whose i-th element is ||P_i||^2
        """
        # v = np.power( norm(P, axis=1), 2 )
        v = np.apply_along_axis(lambda row: np.power(norm(row, ord=2),2), 1, P)
        reward = -0.5* pi.dot(v)
        
        return reward


    def calc_value(self, pi):
        """
        Input:
        pi - population distribution as a row vector

        Returns V(pi; w) = varphi(pi) dot self.w
        where varphi(pi) is the feature vector constructed using pi
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
        # calculate value by inner product
        value = np.array(list_features).dot(self.w)

        # This pure numpy version is actually much slower
#        array_tuples = np.vstack(itertools.combinations_with_replacement(pi, 2))
#        # calculate products, axis is vertical
#        array_features = np.apply_along_axis(lambda x: x[0]*x[1], axis=1, arr=array_tuples)
#        # append first-order features along with bias
#        array_features = np.concatenate([array_features, pi, [1]], axis=0)
#        # calculate value by inner product
#        value = array_features.dot(self.w)

        return value


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
        # Create B matrix, whose (i,j) element is
        # B_{ij} = ( -psi(alpha^i_j) + psi(\sum_j alpha^i_j) + log(P_{ij}))
        #             * 2 * <phi(i,j,pi) , theta>

        # (i,j) element of mat1 is psi(alpha^i_j)
        mat1 = special.digamma(self.mat_alpha)
        # Each row of mat2 has same value along the row
        # each element in row i is psi(\sum_j alpha^i_j)
        mat2 = special.digamma( np.ones([self.d, self.d]) * np.sum(self.mat_alpha, axis=1).reshape(self.d, 1) )
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
    

    def calc_gradient_basic(self, P, pi):
        """ 
        Do not use this version
        """
        gradient = 0
        for i in range(self.d):
            for j in range(self.d):
                gradient = gradient - special.digamma(self.mat_alpha[i,j]) * self.mat_alpha_deriv[i,j]

            multiplier = special.digamma( np.sum(self.mat_alpha[i]) )
            for j in range(self.d):
                gradient = gradient + multiplier * self.mat_alpha_deriv[i,j]

            for j in range(self.d):
                gradient = gradient + np.log(P[i,j]) * self.mat_alpha_deriv[i,j]

        return gradient
    
    def calc_gradient(self, P, pi):
        """
        Input:
        P - transition probability matrix
        pi - population distribution as a row vector

        Calculates \nabla_{theta} log (F(P, pi, theta))
        where F is the product of d d-dimensional Dirichlet distributions
        tensor_phi and mat_alpha are global variables computed in sample_action()

        Do not use this version. Use calc_gradient_vectorized()
        """
        # initialize gradient as column vector
        gradient = 0

        for i in range(self.d):

            # psi(\sum_j alpha^i_j)
            multiplier = special.digamma( np.sum(self.mat_alpha[i]) )

            for j in range(self.d):
                
                # first term = - \nabla log(\Gamma(\alpha^i_j))
                # = - psi(alpha^i_j) * 2 * (phi(i,j,pi) dot theta) phi(i,j,pi)
                first_term = - special.digamma(self.mat_alpha[i,j])

                # second term = psi(\sum_j alpha^i_j) * \nabla \alpha^i_j
                # = psi(\sum_j \alpha^i_j) * 2 * (phi(i,j,pi) dot theta) phi(i,j,pi)
                second_term = multiplier

                # third term = \nabla (\alpha^i_j - 1) log(P_{ij})
                # = 2 * (phi(i,j,pi) dot theta) phi(i,j,pi) * log(P_{ij})
                third_term = np.log( P[i,j] )

                gradient = gradient + (first_term + second_term + third_term)*self.mat_alpha_deriv[i,j]

        return gradient


    def train_log(self, vector, filename, str_format):
        f = open(filename, 'a')
        vector.tofile(f, sep=',', format=str_format)
        f.write("\n")
        f.close()
    

    def train(self, num_episodes=4000, gamma=1, constant=0, lr_critic=0.1, lr_actor=0.001, consecutive=100, file_theta='results_syn/theta.csv', file_pi='results_syn/pi.csv', file_reward='results_syn/reward.csv', file_w='results_syn/w.csv', write_file=0, write_all=0):
        """
        Input:
        1. num_episodes - each episode is 16 steps (9am to 12midnight)
        2. gamma - temporal discount
        3. lr_critic - learning rate for value function parameter update
        4. lr_actor - learning rate for policy parameter update
        5. consecutive - number of consecutive episodes for each reporting of average reward

        Main actor-critic training procedure that improves theta and w
        """

        # initialize collection of start states
        self.init_pi0(path_to_dir=os.getcwd()+'/train_normalized')
        self.num_start_samples = self.mat_pi0.shape[0] # number of rows

        list_reward = []
        for episode in range(num_episodes):
            # print("Episode", episode)
            if write_all:
                with open('temp.csv', 'a') as f:
                    f.write('Episode %d \n\n' % episode)
            # Sample starting pi^0 from mat_pi0
            idx_row = np.random.randint(self.num_start_samples)
            # idx_row = 0 # for testing purposes, select the first row of day 1 always
            pi = self.mat_pi0[idx_row, :] # row vector

            discount = 1
            total_reward = 0
            num_steps = 0

            # Stop after finishing the iteration when num_steps=15, because
            # at that point pi_next = the predicted distribution at midnight
            while num_steps < 15:
                num_steps += 1

#                print("pi\n", pi)
#                print(num_steps)
#                print(self.theta)

                # Sample action
                P = self.sample_action(pi)

                if write_all:
                    with open('temp.csv','ab') as f:
                        np.savetxt(f, np.array(['num_steps = %d' % num_steps]), fmt='%s')
                        np.savetxt(f, np.array(['distribution']), fmt='%s')
                        np.savetxt(f, pi.reshape(1, self.d), delimiter=',', fmt='%.6f')
                        np.savetxt(f, np.array(['Action']), fmt='%s')
                        np.savetxt(f, P, delimiter=',', fmt='%.3f')
            
                # Take action, get pi^{n+1} = P^T pi
                pi_next = np.transpose(P).dot(pi)

                reward = self.calc_reward(P, pi, self.d)
                
                # Calculate TD error
                vec_features_next = self.calc_features(pi_next)
                vec_features = self.calc_features(pi)
                # Consider using the terminal condition V^N = 0
                delta = reward + gamma*(vec_features_next.dot(self.w)) - (vec_features.dot(self.w))

                # Update value function parameter
                # w <- w + alpha * delta * varphi(pi)
                # still a column vector
                length = len(vec_features)
                if constant == 1:
                    self.w = self.w + lr_critic * delta * vec_features.reshape(length,1)
                else:
                    self.w = self.w + (lr_critic/(episode+1)) * delta * vec_features.reshape(length,1)

                # theta update
                gradient = self.calc_gradient_vectorized(P, pi)
                if constant == 1:
                    # NOTE THE SIGN IS NOW POSITIVE (mfg_ac2 has negative)
                    self.theta = self.theta + lr_actor * delta * gradient 
                else:
                    # NOTE THE SIGN IS NOW POSITIVE (mfg_ac2 has negative)
                    self.theta = self.theta + (lr_actor/((episode+1)*np.log(np.log(episode+20)))) * delta * gradient

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
                    self.train_log(self.w, file_w, "%.5e")


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
        P[P<=0] = 1e-100
        Q[Q<=0] = 1e-100

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
        1. mat_trajectory, each row is the distribution at a discrete time step,
        from pi^0 to pi^N
        2. array_actions - each element is an entire transition matrix P
        """

        pi = pi0
        # Initialize matrix to store trajectory
        # total_steps rows by d columns
        mat_trajectory = np.zeros([total_hours, self.d])
        # Store initial distribution
        mat_trajectory[0] = pi
        array_actions = np.zeros([total_hours-1, self.d, self.d])
        hour = 1

        while hour < total_hours:
            P = self.sample_action(pi)
            array_actions[hour-1] = P
            pi_next = np.transpose(P).dot(pi)
            mat_trajectory[hour] = pi_next
            pi = pi_next
            hour += 1

        return mat_trajectory, array_actions


    def evaluate(self, theta=7.401786, d=21, episode_length=16, indir='test_normalized', outfile='test_eval.csv'):
        """
        Main evaluation function

        Argument:
        theta - value to use for the fixed policy
        indir - directory containing the test dataset

        """
        # Fix policy by setting parameter
        self.theta = theta
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
            mat_trajectory, array_actions = self.generate_trajectory(pi0, episode_length)

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
        mean_l1_mean = np.mean(array_l1_mean)
        mean_JSD_final = np.mean(array_JSD_final)
        mean_JSD_mean = np.mean(array_JSD_mean)

        with open(outfile, 'ab') as f:
            np.savetxt(f, np.array(['theta = %f' % self.theta]), fmt='%s')
            np.savetxt(f, np.array(['array_l1_final']), fmt='%s')
            np.savetxt(f, np.array([mean_l1_final]), fmt='%.3e')
            np.savetxt(f, array_l1_final.reshape(1, num_test_trajectories), delimiter=',', fmt='%.3e')
            np.savetxt(f, np.array(['array_l1_mean']), fmt='%s')
            np.savetxt(f, np.array([mean_l1_mean]), fmt='%.3e')
            np.savetxt(f, array_l1_mean.reshape(1, num_test_trajectories), delimiter=',', fmt='%.3e')
            np.savetxt(f, np.array(['array_JSD_final']), fmt='%s')
            np.savetxt(f, np.array([mean_JSD_final]), fmt='%.3e')
            np.savetxt(f, array_JSD_final.reshape(1, num_test_trajectories), delimiter=',', fmt='%.3e')
            np.savetxt(f, np.array(['array_JSD_mean']), fmt='%s')
            np.savetxt(f, np.array([mean_JSD_mean]), fmt='%.3e')
            np.savetxt(f, array_JSD_mean.reshape(1, num_test_trajectories), delimiter=',', fmt='%.3e')

        print("mat_trajectory\n", mat_trajectory)
        print("array_l1_final\n", array_l1_final)
        print("array_l1_mean\n", array_l1_mean)
        print("array_JSD_final\n", array_JSD_final)
        print("array_JSD_mean\n", array_JSD_mean)


    def visualize(self, theta=8.86349, d=21, topic=0, dir_train='train_normalized', train_start=1, train_end=27, dir_test='test_normalized', test_start=27, test_end=38):
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
        for num_day in range(train_start, train_end):
            # Read initial distribution pi0
            pi0 = np.array(df_train.iloc[(num_day-1)*16])

            # Generate entire trajectory using policy
            mat_trajectory, array_actions = self.generate_trajectory(pi0, total_hours=16)
            df = pd.DataFrame(mat_trajectory)
            df.index = np.arange(idx, idx+16)
            list_df.append(df)
            idx += 16

        self.df_train_generated = pd.concat(list_df)
        self.df_train_generated.index = pd.to_datetime(self.df_train_generated.index, unit="D")

        # Generate trajectory using policy on test data
        print("Generating trajectory from test data")
        list_df = []
        for num_day in range(test_start, test_end):
            # Read initial distribution
            pi0 = np.array(df_test.iloc[(num_day-test_start)*16])
            # Generate entire trajectory using policy
            mat_trajectory, array_actions = self.generate_trajectory(pi0, total_hours=16)
            df = pd.DataFrame(mat_trajectory)
            df.index = np.arange(idx, idx+16) # use same idx that was incremented above
            list_df.append(df)
            idx += 16
        self.df_test_generated = pd.concat(list_df)
        self.df_test_generated.index = pd.to_datetime(self.df_test_generated.index, unit="D")

        num_train = len(self.df_train_generated.index)
        array_x_train = np.arange(num_train)
        array_x_test = np.arange(num_train, num_train+len(self.df_test_generated.index))

        plt.plot(array_x_train, df_train[topic], color='r', linestyle='-', label='train data')
        plt.plot(array_x_train, self.df_train_generated[topic], color='b', linestyle='--', label='MFG (train)')
        plt.plot(array_x_test, df_test[topic], color='k', linestyle='-', label='test data')
        plt.plot(array_x_test, self.df_test_generated[topic], color='g', linestyle='--', label='MFG (test)')        
        plt.ylabel('Topic %d popularity' % topic)
        plt.xlabel('Time steps (hrs)')
        plt.legend(loc='best')
        plt.title("Topic %d empirical and generated data" % topic)
        plt.show()
        

    def calc_reward_vector(self, P):
        """
        Input:
        P - transition matrix

        Using r_i(pi, P_i) = -1/2 ||P_i||^2, 
        reward vector is v
        where v is vector whose i-th element is -1/2 ||P_i||^2
        """
        # v = -0.5 * np.power( norm(P, axis=1), 2 )
        v = -0.5 * np.apply_along_axis(lambda row: np.power( norm(row, ord=2), 2), 1, P)

        return v


    def evaluate_synthetic(self, day_first=1, day_last=26, verbose=0):
        """
        Evaluates how close P_{ij}^n is to
        1. V_j^n - V_i^n , if i \neq j
        2. - \sum_{j: j \neq i} (V_j^n - V_i^n) + 1 , if i == j

        Argument:
        day_first - first training file
        day_last - last training file
        verbose - obvious

        Return:
        diff_mean
        diff_std

        For each day, for each hour, sum up the 
        absolute difference between P_{ij} and the value
        computed using the value function, over all i and all j
        This gives an array of values, one value for each hour, for all hours and all days.
        Report the mean and standard deviation of this array.

        Requires self.theta to be set prior to running this. 
        """

        list_diff = []
        # For each initial distribution, use policy to generate trajectory
        # pi^0, P^0, pi^1, P^1,...pi^N
        for day in range(day_first-1, day_last):
            # Get initial distribution
            pi = self.mat_pi0[day, :]
            # Get sequence of distributions pi^n and actions P^n
            mat_trajectory, array_actions = self.generate_trajectory(pi, total_hours=16)

            # Use backward equation to get V^n_i for all n, for all i
            # num_topic rows, 16 columns, each column is V^n for that hour
            mat_V = np.zeros([self.d, 16]) 
            # From 14 to 0, inclusive
            for n in range(14,-1,-1):
                vec_reward = self.calc_reward_vector( array_actions[n] )
                # V^n = r + P * V^{n+1}
                mat_V[: , n:n+1] = vec_reward.reshape(self.d, 1) + array_actions[n].dot( mat_V[:, n+1:n+2] )
                
            # Go through all P matrices for this day,
            # get sum of absolute difference between P_{ij} and the value computed
            # using the value function, for all i for all j
            for n in range(0,15):
                diff = 0
                for idx_i in range(0, self.d):
                    for idx_j in range(0, self.d):
                        P_ij = array_actions[n, idx_i, idx_j]
                        if idx_i == idx_j:
                            # (-\sum_{j: j != i} V_j - V_i) + 1
                            value = 1 - (np.sum(mat_V[:, n]) - self.d*mat_V[idx_i, n])
                        else:
                            # V_j - V_i
                            value = mat_V[idx_j, n] - mat_V[idx_i, n]
                        diff += abs(P_ij - value)
                list_diff.append(diff)

        diff_mean = np.mean(list_diff)
        diff_std = np.std(list_diff)

        if verbose:
            print("Mean over all hours", diff_mean)
            print("Standard deviation", diff_std)
                    
            print("P_01 at time 0 =", array_actions[0,0,1])
            print("V_1 - V_0 at time 0 =", (mat_V[1,0] - mat_V[0,0]))
    
            print("P_00 at time 0 =", array_actions[0,0,0])
            print("value =", (1 - (np.sum(mat_V[:,0]) - mat_V[0,0] - (self.d-1)*mat_V[0,0])))
        return diff_mean, diff_std


    def evaluate_synthetic_JSD(self, day_first=1, day_last=26, write_file=0, filename='synthetic_log.csv', verbose=0):
        """
        Evaluates how close P_i^n is to the row constructed from
        1. V_j^n - V_i^n , if i \neq j
        2. - \sum_{j: j \neq i} (V_j^n - V_i^n) + 1 , if i == j

        Argument:
        day_first - first training file
        day_last - last training file
        write_file - set to 1 to write file
        filename - name of output file
        verbose - obvious

        Return:
        diff_mean
        diff_std

        For each day, for each hour, sum up the 
        JSD between P_i and the value computed using the value function
        over all i
        This gives an array of values, one value for each hour, for all hours and all days.
        Report the mean and standard deviation of this array.

        Requires self.theta to be set prior to running this. 
        """

        list_diff = []
        # For each initial distribution, use policy to generate trajectory
        # pi^0, P^0, pi^1, P^1,...pi^N
        for day in range(day_first-1, day_last):
            # Get initial distribution
            pi = self.mat_pi0[day, :]
            # Get sequence of distributions pi^n and actions P^n
            mat_trajectory, array_actions = self.generate_trajectory(pi, total_hours=16)

            # Use backward equation to get V^n_i for all n, for all i
            # num_topic rows, 16 columns, each column is V^n for that hour
            mat_V = np.zeros([self.d, 16]) 
            # From 14 to 0, inclusive
            for n in range(14,-1,-1):
                vec_reward = self.calc_reward_vector( array_actions[n] )
                # V^n = r + P * V^{n+1}
                mat_V[: , n:n+1] = vec_reward.reshape(self.d, 1) + array_actions[n].dot( mat_V[:, n+1:n+2] )
                
            # Go through all P matrices for this day,
            # get sum of JSD between P_i and the row computed
            # using the value function, for all row i
            for n in range(0,15):
                diff = 0
                # Go through all rows
                for idx_i in range(0, self.d):
                    P_i = array_actions[n, idx_i]
                    # Compute the row for comparison
                    row_compare = np.zeros(self.d)
                    for idx_j in range(0, self.d):
                        if idx_i == idx_j:
                            # (-\sum_{j: j != i} V_j - V_i) + 1
                            row_compare[idx_j] = 1 - (np.sum(mat_V[:, n]) - self.d*mat_V[idx_i, n])
                        else:
                            # V_j - V_i
                            row_compare[idx_j] = mat_V[idx_j, n] - mat_V[idx_i, n]
                    diff += self.JSD(P_i, row_compare)
                if write_file:
                    self.train_log(P_i, filename, "%.3e")
                    self.train_log(row_compare, filename, "%.3e")
                    self.train_log(np.array([0]), filename, "%d") # line seperator
#                    if diff == np.inf:
#                        break
                list_diff.append(diff)
#            print(P_i)
#            print(row_compare)
#        print(list_diff)
        diff_mean = np.mean(list_diff)
        diff_std = np.std(list_diff)

        if verbose:
            print("Mean over all hours", diff_mean)
            print("Standard deviation", diff_std)
                    
            print("P_01 at time 0 =", array_actions[0,0,1])
            print("V_1 - V_0 at time 0 =", (mat_V[1,0] - mat_V[0,0]))
    
            print("P_00 at time 0 =", array_actions[0,0,0])
            print("value =", (1 - (np.sum(mat_V[:,0]) - mat_V[0,0] - (self.d-1)*mat_V[0,0])))
        return diff_mean, diff_std    


if __name__ == "__main__":
    # Try to find a good theta (current best is 2.6)
    with open("synthetic.csv", 'a') as f:
        f.write("Shift,theta_initial,theta_final,diff_mean,diff_std\n")
    
    for shift in np.arange(0, 0.04, 0.02):
        for theta_initial in np.arange(0, 5.0, 0.05):
            print("Theta initial", theta_initial)
            print("Shift", shift)
            ac = actor_critic(theta=theta_initial, shift=shift, alpha_scale=10000, d=21)
        #ac = actor_critic(theta=theta, shift=0, alpha_scale=10000, d=21)
            try:
                ac.train(num_episodes=1000, gamma=1, constant=1, lr_critic=0.1, lr_actor=0.001, consecutive=100, write_file=0)
            except:
                pass
            print("Theta final is", ac.theta)
        #ac.init_pi0('train_normalized')
            try:
                diff_mean, diff_std = ac.evaluate_synthetic_JSD(day_first=1, day_last=26)
            except:
                diff_mean = 900
                diff_std = 900
            with open("synthetic.csv", 'a') as f:
                f.write("%.3f,%.3f,%.3f,%.3f,%.3f\n" % (shift, theta_initial, ac.theta, diff_mean, diff_std))
