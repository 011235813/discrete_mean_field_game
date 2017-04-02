import numpy as np
import os
from scipy import special

class actor_critic:

    def __init__(self, dim_theta=6, d=40, num_episodes=4000):

        self.dim_theta = dim_theta
        # initialize theta as random column vector, entries [0,1)
        self.theta = np.random.rand(dim_theta, 1)

        # initialize weight vector for value function approximation
        self.w = self.init_w()

        self.num_episodes = num_episodes

        # initialize collection of start states
        self.init_pi0(path_to_dir=r'C:\Users\Jiachen\Documents\Projects\Python\RL\MFG\data_train_reordered')
        self.num_start_samples = self.mat_pi0.shape[0] # number of rows

        # number of topics
        self.d = d
        
    def init_w(self, dim_w=50):
        """
        Initialize weight vector for value function approximation
        Two choices:
        1. Choose dimension of this vector depending on the minimum length
        of \pi^0, because the number of trending topics retrieved might vary per day
        (e.g. some days have 50, some have 48)
        2. Ensure that all \pi^0 have the same number of topics, then no need to
        find minimum
        Also decide whether to include the null topic
        """
        return np.random.rand(dim_w, 1)


    def init_pi0(self, path_to_dir='/home/t3500/devdata/mfg/distribution/train_reordered'):
        """
        Generates the collection of initial population distributions.
        This collection will be sampled to get the start state for each training episode
        Assumes that each file in director has rows of the format:
        pi^0_1, ... , pi^0_d
        where d is a fixed constant across all files
        """
        list_pi0 = []
        for filename in os.listdir(path_to_dir):
            path_to_file = path_to_dir + '/' + filename
            f = open(path_to_file, 'r')
            list_lines = f.readlines()
            f.close()

            list_pi0.append( list(map(int, list_lines[0].strip().split(','))) )
            
        num_rows = len(list_pi0)
        num_cols = len(list_pi0[0])

        self.mat_pi0 = np.zeros([num_rows, num_cols])
        for i in range(len(list_pi0)):
            self.mat_pi0[i] = list_pi0[i]
        

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

    
    def reorder_files(self, path_to_dir='/home/t3500/devdata/mfg/distribution/train', output_dir='/home/t3500/devdata/mfg/distribution/train_reordered'):
        """
        Process all files in given directory, creates new files
        """
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
            f = open(output_dir + '/' + filename_new, 'w')
            for row in list_rows:
                s = ','.join(map(str, row))
                s += '\n'
                f.write(s)
            f.close()


    def sample_action(self, pi):
        """
        Samples from product of d d-dimensional Dirichlet distributions
        Input:
        pi - row vector
        Returns an entire transition probability matrix
        """
        # Construct all alphas
        mat_alpha = np.zeros([self.d, self.d])
        for i in range(self.d):
            # d x (num_features) matrix
            mat_phi = np.zeros([self.d, self.dim_theta])
            # each row is phi(i, j, pi)
            for j in range(self.d):
                # construct feature vector, (num_features) x 1
                phi = [1, pi[i], pi[j], pi[i]*pi[j], pi[i]**2, pi[j]**2]
                # insert into mat_phi
                mat_phi[j] = phi
            temp = mat_phi.dot(self.theta) # d x 1
            # element-wise product, to get all entries nonzero
            alpha = temp * temp # d x 1
            # Insert check for zero
            for element in alpha:
                if element <= 0:
                    print("Error! element of alpha is non-positive!")
            # Insert alpha transpose into mat_alpha as the i-th row
            mat_alpha[i] = np.transpose(alpha)
        
        # Sample matrix P from Dirichlet
        P = np.zeros([self.d, self.d])
        for i in range(self.d):
            # Get y^i_1, ... y^i_d
            y = [np.random.gamma(shape=a, scale=1) for a in mat_alpha[i, :]]
            total = np.sum(y)
            # Store into i-th row of matrix P
            P[i] = [y_j/total for y_j in y]

        return P
            

    def train(self, gamma=0.99, lr_critic=0.2, lr_actor=0.6):
        """
        Main actor-critic training procedure that improves theta and w
        """
        for episode in range(self.num_episodes):

            # Sample starting pi^0 from mat_pi0
            idx_row = np.random.randint(self.num_start_samples)
            pi = self.mat_pi0[idx_row, :] # row vector

            discount = 1
            total_cost = 0
            num_steps = 0

            while num_steps < 16:
                num_steps += 1

                # Sample action
                P = self.sample_action(pi)
            
                # Incomplete
            
