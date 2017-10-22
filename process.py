import os
import numpy as np

def combine_files(start=1, end=21, read_dir='train_normalized_round2', write_location='rnn_round2/rnn_train.txt'):
    
    f_out = open(write_location, 'w')

    for idx in range(start, end+1):
        filename = read_dir + "/trend_distribution_day%d.csv" % idx
        with open(filename, 'r') as f:
            matrix = np.loadtxt(f, delimiter=' ')
        matrix = matrix[:, 0:15]
        s = ''
        for hour in range(0,16):
            s += ','.join(map(str, matrix[hour]))
            if hour < 15:
                s += ' '
            elif hour == 15:
                s += '\n'
        f_out.write(s)
        
    f_out.close()

                
        
        
