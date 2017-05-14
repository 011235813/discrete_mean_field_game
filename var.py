import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
import matplotlib.pylab as plt
import os
from numpy.linalg import norm
from scipy.stats import entropy

from statsmodels.tsa.base.datetools import dates_from_str
from statsmodels.tsa.stattools import adfuller

class var():

    #def __init__(self, train='train_normalized', test='test_normalized', d=21):
    def __init__(self, d=21):
        """
        Arguments:
        d - number to topics to use (includes the null topic at index 0)
        """
        # self.df_train, self.df_test = self.read_data(train, test, d)
        self.d = d
#        self.mdata = sm.datasets.macrodata.load_pandas().data
#
#        self.dates = self.mdata[['year', 'quarter']].astype(int).astype(str)
#
#        self.quarterly = self.dates["year"] + "Q" + self.dates["quarter"]
#
#        self.quarterly = dates_from_str(self.quarterly)
#
#        self.mdata = self.mdata[['realgdp', 'realcons', 'realinv']]
#
#        self.mdata.index = pd.DatetimeIndex(self.quarterly)
#
#        self.data = np.log(self.mdata).diff().dropna()
#
#        self.model = VAR(self.data)
#
#        self.results = self.model.fit(2)


    def read_data(self, train='train_normalized2', train_start=1, train_end=35, test='test_normalized2', test_start=36, test_end=45):
        """
        Arguments:
        train - directory that holds normalized training data
        train_start - the smallest day number among training files
        train_end - the largest day number among training files
        test - directory that holds normalized test data
        test_start - the smallest day number among test files
        test_end - the largest day number among test files

        """
        print("Reading train files")
        list_df = []
        idx = 0
        # for filename in os.listdir(train):
        for num_day in range(train_start, train_end+1):
            filename = "trend_distribution_day%d_reordered.csv" % num_day
            print(filename)
            path_to_file = train + '/' + filename
            df = pd.read_csv(path_to_file, sep=' ', header=None, names=range(self.d), usecols=range(self.d), dtype=np.float64)
            df.index = np.arange(idx, idx+16)
            list_df.append(df)
            idx += 16
            
        df_train = pd.concat(list_df)
        df_train.index = pd.to_datetime(df_train.index, unit="D")
        self.df_train = df_train
        
        print("Reading test files")
        list_df = []
        # for filename in os.listdir(test):
        for num_day in range(test_start, test_end+1):
            filename = "trend_distribution_day%d_reordered.csv" % num_day
            print(filename)
            path_to_file = test + '/' + filename
            df = pd.read_csv(path_to_file, sep=' ', header=None, names=range(self.d), usecols=range(self.d), dtype=np.float64)
            df.index = np.arange(idx, idx+16) # use the same idx that was incremented when reading training data
            list_df.append(df)
            idx += 16
        if len(list_df):
            df_test = pd.concat(list_df)
        else:
            df_test = pd.DataFrame()

        df_test.index = pd.to_datetime(df_test.index, unit="D")
        self.df_test = df_test
        
        return self.df_train, self.df_test


    def check_stationarity(self, topic):
        ts = self.df_train[topic]

        #Determing rolling statistics
        rolmean = pd.rolling_mean(ts, window=12)
        rolstd = pd.rolling_std(ts, window=12)
    
        #Plot rolling statistics:
        orig = plt.plot(ts, color='blue',label='Original')
        mean = plt.plot(rolmean, color='red', label='Rolling Mean')
        std = plt.plot(rolstd, color='black', label = 'Rolling Std')
        plt.legend(loc='best')
        plt.title('Rolling Mean & Standard Deviation')
        plt.show(block=False)
        
        #Perform Dickey-Fuller test:
        print('Results of Dickey-Fuller Test:')
        dftest = adfuller(ts, autolag='AIC')
        dfoutput = pd.Series(dftest[0:4], index=['Test Statistic','p-value','#Lags Used','Number of Observations Used'])
        for key,value in dftest[4].items():
            dfoutput['Critical Value (%s)'%key] = value
        print(dfoutput)

        
    def train(self, max_lag, df_train):

        self.model = VAR(df_train)

        self.results = self.model.fit(maxlags=max_lag, ic='aic')


    def cross_validation(self, lag_range=range(1,21), validation_size=10, repetitions=5):
        """
        Arguments:
        lag_range - the range of lag values to try
        validation_size - number of trajectories to use as the validation set
        """
        list_error = []
        num_training_points = int(len(self.df_train.index) / 16)
        list_choices = range(num_training_points)
        num_selected = num_training_points - validation_size
        # For each lag value
        for lag in lag_range:
            avg_error = 0
            # Repeat for repetition times
            for rep in range(0, repetitions):
                # Randomly split into training set and validation set
                selected = np.random.choice(list_choices, num_selected, replace=False)
                the_rest = [x for x in list_choices if x not in selected]
                list_temp = []
                for point in selected:
                    list_temp.append( self.df_train[point:point+16] )
                df_selected = pd.concat(list_temp)
                list_temp = []
                for point in the_rest:
                    list_temp.append( self.df_train[point:point+16] )
                df_validation = pd.concat(list_temp)

                # Relabel indices to have increasing time order
                df_selected.index = np.arange(len(df_selected.index))
                df_selected.index = pd.to_datetime(df_selected.index, unit="D")
                df_validation.index = np.arange(len(df_selected.index), len(df_selected.index) + 16*validation_size)
                df_validation.index = pd.to_datetime(df_validation.index, unit="D")
                # Train
                self.train(max_lag=lag, df_train=df_selected)

                # Test on the validation set and accumulate error                
                avg_error += self.validation(validation_size*16, df_selected, df_validation)

            # Average error over repetitions
            avg_error = avg_error / repetitions
            print("Lag %d. avg_error" % lag, avg_error)
            # Record avg error for this lag value
            list_error.append(avg_error)

        print("Min error is", np.min(list_error))
        print("Best lag value is", lag_range[np.argmin(list_error)])
        f = open('var_cross_val.csv', 'a')
        s = ','.join(map(str, lag_range))
        s += '\n'
        f.write(s)
        s = ','.join(map(str, list_error))
        s += '\n'
        f.write(s)


    def plot(self, topic, lag):
        
        #plt.plot(self.df_train.index[lag:], self.df_train[topic][lag:], color='r', label='data')
        plt.plot(self.df_train.index, self.df_train[topic], color='r', label='data')        
        plt.plot(self.df_train.index[lag:], self.results.fittedvalues[topic], color='b', label='time series')
        plt.legend(loc='best')
        plt.title('Topic %d data and fitted time series' % topic)
        plt.show()
        

    def JSD(self, P, Q):
        """
        Arguments:
        P,Q - discrete probability distribution
        
        Return:
        Jensen-Shannon divergence
        """

        # Replace all invalid values by 1e-100
        P[P<=0] = 1e-100
        Q[Q<=0] = 1e-100

        P_normed = P / norm(P, ord=1)
        Q_normed = Q / norm(Q, ord=1)
        M = 0.5 * (P + Q)

        return 0.5 * (entropy(P,M) + entropy(Q,M))


    def evaluate_train(self):
        lag = self.results.k_ar

        # Total number of distributions in fitted time series
        len_fitted = len(self.results.fittedvalues.index)
        # Total number of distributions across all days and all hours in training set
        len_empirical = len(self.df_train.index)

        ### Part 1: evaluate final distributions only ###

        # index of fittedvalues that corresponds to the final distribution on day1
        # e.g. if lag is 15, then the index 0 of fittedvalues is the final distribution of day1
        idx_fitted = 15 - lag
        # start at final distribution of day1
        idx_empirical = 15

        # num_trajectories = number of days
        num_trajectories = int(len_empirical/16)
        array_l1_final = np.zeros(num_trajectories)
        array_JSD_final = np.zeros(num_trajectories)
        
        # Go through all final distributions
        idx =  0
        while idx_fitted < len_fitted and idx_empirical < len_empirical:
            l1_final = norm( self.df_train.ix[idx_empirical] - self.results.fittedvalues.ix[idx_fitted], ord=1)
            array_l1_final[idx] = l1_final

            JSD_final = self.JSD( self.df_train.ix[idx_empirical], self.results.fittedvalues.ix[idx_fitted] )
            array_JSD_final[idx] = JSD_final

            idx_fitted += 16
            idx_empirical += 16
            idx += 1

        ### Part 2: evaluate distributions at all hours ###

        array_l1_mean = np.zeros(len_fitted)
        array_JSD_mean = np.zeros(len_fitted)
        idx_fitted = 0 # start at very beginning

        while idx_fitted < len_fitted:
            l1 = norm( self.df_train.ix[idx_fitted+lag] - self.results.fittedvalues.ix[idx_fitted], ord=1)
            array_l1_mean[idx_fitted] = l1

            JSD_final = self.JSD( self.df_train.ix[idx_fitted+lag], self.results.fittedvalues.ix[idx_fitted] )
            array_JSD_mean[idx_fitted] = JSD_final            
            idx_fitted += 1

        # Mean over all days of the difference between final distributions
        mean_l1_final = np.mean(array_l1_final)
        mean_JSD_final = np.mean(array_JSD_final)
        print(array_l1_final)
        print(array_JSD_final)
        print(mean_l1_final)
        print(mean_JSD_final)

        # Mean over all hours of the difference between distributions at all hours
        mean_l1 = np.mean(array_l1_mean)
        mean_JSD = np.mean(array_JSD_mean)
        print(mean_l1)
        print(mean_JSD)
        

    def validation(self, steps, df_selected, df_validation):
        """
        Arguments:
        steps - number of future points to generate
        df_selected - subset of self.df_train selected for training
        df_validation - subset of self.df_train not selected for training

        Return:
        mean_JSD_mean - mean over all validation days of the mean JSD over all hours
        """
        num_previous = len(df_selected.index)
        lag_order = self.results.k_ar
        future = self.results.forecast(df_selected.values[-lag_order:], steps)
        df_future = pd.DataFrame(future)
        df_future.index = np.arange(num_previous, num_previous+steps)
        df_future.index = pd.to_datetime(df_future.index, unit="D")

        len_validation = len(df_validation.index)
        num_trajectories = int(len_validation / 16)
        array_JSD_mean = np.zeros(num_trajectories)
        idx_day = 0
        while idx_day < num_trajectories:
            idx_hour = 0
            jsd = 0
            while idx_hour < 16:
                idx = 16*idx_day + idx_hour
                jsd += self.JSD( df_validation.ix[idx], df_future.ix[idx] )
                idx_hour += 1

            array_JSD_mean[idx_day] = jsd/16
            idx_day += 1

        mean_JSD_mean = np.mean(array_JSD_mean)
        return mean_JSD_mean


    def forecast(self, num_prior=416, steps=176, topic=0, plot=1, show_plot=1):
        """
        Arguments:
        num_prior - number of training datapoints prior to start of future to use
        steps - number of future points to generate
        topic - topic to plot
        """

        lag = self.results.k_ar
        num_previous = len(self.df_train.index)
        
        future = self.results.forecast(self.df_train.values[-num_prior:], steps)

        self.df_future = pd.DataFrame(future)
        self.df_future.index = np.arange(num_previous, num_previous+steps)
        self.df_future.index = pd.to_datetime(self.df_future.index, unit="D")

        array_x_train = np.arange(num_previous)
        array_x_test = np.arange(num_previous, num_previous+len(self.df_future.index))

        if plot == 1:
            # For plotting future along with raw data and fitted time series
            plt.plot(array_x_train, self.df_train[topic], color='r', linestyle='-', label='train data')        
            plt.plot(array_x_train[lag:], self.results.fittedvalues[topic], color='b', linestyle='--', label='time series (train)')
            plt.plot(array_x_test, self.df_test[topic], color='k', linestyle='-', label='test data')
            plt.plot(array_x_test, self.df_future[topic], color='g', linestyle='--', label='time series (test)')
            plt.ylabel('Topic %d popularity' % topic)
            plt.xlabel('Time steps (hrs)')
            plt.legend(loc='best')
            plt.title('Topic %d data and fitted time series' % topic)
            if show_plot == 1:
                plt.show()

        return self.df_future


    def evaluate_test(self, outfile='test_eval_var.csv'):
        lag = self.results.k_ar

        # Total number of distributions in future 
        len_future = len(self.df_future.index)
        # Total number of distributions across all days and all hours in test set
        len_empirical = len(self.df_test.index)
        
        if len_future != len_empirical:
            print("Lengths of test set and generated future differ!")
            return

        ### Part 1: evaluate final distributions only ###

        # start at final distribution of day1
        idx_future = 15
        idx_empirical = 15

        # num_trajectories = number of days
        num_trajectories = int(len_empirical/16)
        array_l1_final = np.zeros(num_trajectories)
        array_JSD_final = np.zeros(num_trajectories)
        
        # Go through all final distributions
        idx =  0
        while idx_future < len_future and idx_empirical < len_empirical:
            l1_final = norm( self.df_test.ix[idx_empirical] - self.df_future.ix[idx_future], ord=1)
            array_l1_final[idx] = l1_final

            JSD_final = self.JSD( self.df_test.ix[idx_empirical], self.df_future.ix[idx_future] )
            array_JSD_final[idx] = JSD_final

            idx_future += 16
            idx_empirical += 16
            idx += 1

        ### Part 2: evaluate distributions at all hours ###

        array_l1_mean = np.zeros(num_trajectories)
        array_JSD_mean = np.zeros(num_trajectories)
        idx_day = 0
        while idx_day < num_trajectories:
            idx_hour = 0
            l1 = 0
            jsd = 0
            while idx_hour < 16:
                idx = 16*idx_day + idx_hour
                l1 += norm( self.df_test.ix[idx] - self.df_future.ix[idx], ord=1 )
                jsd += self.JSD( self.df_test.ix[idx], self.df_future.ix[idx] )
                idx_hour += 1

            array_l1_mean[idx_day] = l1/16
            array_JSD_mean[idx_day] = jsd/16
            idx_day += 1

        # Mean over all days of the difference between final distributions
        mean_l1_final = np.mean(array_l1_final)
        std_l1_final = np.std(array_l1_final)
        mean_JSD_final = np.mean(array_JSD_final)
        std_JSD_final = np.std(array_JSD_final)
        print(array_l1_final)
        print(array_JSD_final)
        print(mean_l1_final)
        print(mean_JSD_final)

        # Mean over all hours of the difference between distributions at all hours
        mean_l1_mean = np.mean(array_l1_mean)
        std_l1_mean = np.std(array_l1_mean)
        mean_JSD_mean = np.mean(array_JSD_mean)
        std_JSD_mean = np.std(array_JSD_mean)
        print(mean_l1_mean)
        print(mean_JSD_mean)        

        with open(outfile, 'ab') as f:
            np.savetxt(f, np.array(['array_l1_final']), fmt='%s')
            np.savetxt(f, np.array([mean_l1_final]), fmt='%.3e')
            np.savetxt(f, np.array([std_l1_final]), fmt='%.3e')            
            np.savetxt(f, array_l1_final.reshape(1, num_trajectories), delimiter=',', fmt='%.3e')
            np.savetxt(f, np.array(['array_l1_mean']), fmt='%s')
            np.savetxt(f, np.array([mean_l1_mean]), fmt='%.3e')
            np.savetxt(f, np.array([std_l1_mean]), fmt='%.3e')            
            np.savetxt(f, array_l1_mean.reshape(1, num_trajectories), delimiter=',', fmt='%.3e')
            np.savetxt(f, np.array(['array_JSD_final']), fmt='%s')
            np.savetxt(f, np.array([mean_JSD_final]), fmt='%.3e')
            np.savetxt(f, np.array([std_JSD_final]), fmt='%.3e')
            np.savetxt(f, array_JSD_final.reshape(1, num_trajectories), delimiter=',', fmt='%.3e')
            np.savetxt(f, np.array(['array_JSD_mean']), fmt='%s')
            np.savetxt(f, np.array([mean_JSD_mean]), fmt='%.3e')
            np.savetxt(f, np.array([std_JSD_mean]), fmt='%.3e')
            np.savetxt(f, array_JSD_mean.reshape(1, num_trajectories), delimiter=',', fmt='%.3e')



if __name__ == "__main__":
    exp = var(d=21)
    print("reading data")
    exp.read_data()
    print("training")
    exp.train(max_lag=15, df_train=self.df_train)
    print("evaluate training performance")
    exp.evaluate_train()
    print("forecasting")
    exp.forecast()
    print("evaluate test performance")
    exp.evaluate_test()
