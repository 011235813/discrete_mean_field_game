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


    def read_data(self, train='train_normalized', train_start=1, train_end=27, test='test_normalized', test_start=27, test_end=38):
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
        for num_day in range(train_start, train_end):
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
        for num_day in range(test_start, test_end):
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

        
    def train(self, max_lag=15):

        self.model = VAR(self.df_train)

        self.results = self.model.fit(maxlags=max_lag, ic='aic')


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
        

    def forecast(self, num_prior=416, steps=176, topic=0):
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

        # For plotting future along with raw data and fitted time series
        plt.plot(array_x_train, self.df_train[topic], color='r', linestyle='-', label='train data')        
        plt.plot(array_x_train[lag:], self.results.fittedvalues[topic], color='b', linestyle='--', label='time series (train)')
        plt.plot(array_x_test, self.df_test[topic], color='k', linestyle='-', label='test data')
        plt.plot(array_x_test, self.df_future[topic], color='g', linestyle='--', label='time series (test)')
        plt.ylabel('Topic %d popularity' % topic)
        plt.xlabel('Time steps (hrs)')
        plt.legend(loc='best')
        plt.title('Topic %d data and fitted time series' % topic)
        plt.show()


    def evaluate_test(self):
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

        array_l1_mean = np.zeros(len_future)
        array_JSD_mean = np.zeros(len_future)
        idx_future = 0 # start at very beginning

        while idx_future < len_future:
            l1 = norm( self.df_test.ix[idx_future] - self.df_future.ix[idx_future], ord=1)
            array_l1_mean[idx_future] = l1

            JSD_final = self.JSD( self.df_test.ix[idx_future], self.df_future.ix[idx_future] )
            array_JSD_mean[idx_future] = JSD_final            
            idx_future += 1

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


if __name__ == "__main__":
    exp = var(d=21)
    print("reading data")
    exp.read_data()
    print("training")
    exp.train()
    print("evaluate training performance")
    exp.evaluate_train()
    print("forecasting")
    exp.forecast()
    print("evaluate test performance")
    exp.evaluate_test()
