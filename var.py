import numpy as np
import pandas as pd
import statsmodels.api as sm
from statsmodels.tsa.api import VAR, DynamicVAR
import matplotlib.pylab as plt
import os

from statsmodels.tsa.base.datetools import dates_from_str
from statsmodels.tsa.stattools import adfuller

class var():

    def __init__(self, train='train_normalized', test='test_normalized', d=21):
        """
        Arguments:
        d - number of topics
        """
        self.df_train, self.df_test = self.read_data(train, test, d)

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


    def read_data(self, train, test, d):

        print("Reading train files")
        list_df = []
        idx = 0
        for filename in os.listdir(train):
            print(filename)
            path_to_file = train + '/' + filename
            df = pd.read_csv(path_to_file, sep=' ', header=None, names=range(d), usecols=range(d), dtype=np.float64)
            df.index = np.arange(idx, idx+16)
            list_df.append(df)
            idx += 16
            
        df_train = pd.concat(list_df)
        df_train.index = pd.to_datetime(df_train.index, unit="D")

        print("Reading test files")
        list_df = []
        idx = 0
        for filename in os.listdir(test):
            print(filename)
            path_to_file = test + '/' + filename
            df = pd.read_csv(path_to_file, sep=' ', header=None, names=range(d), usecols=range(d), dtype=np.float64)
            df.index = np.arange(idx, idx+16)
            list_df.append(df)
            idx += 16
        if len(list_df):
            df_test = pd.concat(list_df)
        else:
            df_test = pd.DataFrame()

        df_test.index = pd.to_datetime(df_test.index, unit="D")

        return df_train, df_test


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
        
        plt.plot(self.df_train.index[lag:], self.df_train[topic][lag:], color='r', label='data')
        plt.plot(self.df_train.index[lag:], self.results.fittedvalues[topic], color='b', label='time series')
        plt.legend(loc='best')
        plt.title('Topic %d data and fitted time series' % topic)
        plt.show()
        
