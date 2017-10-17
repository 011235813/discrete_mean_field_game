import os
import numpy as np

def combine_files(start=36, end=45):
    
    f_out = open("rnn_test.txt", 'w')

    for idx in range(start, end+1):
        filename = "test_normalized2/trend_distribution_day%d_reordered.csv" % idx
        with open(filename, 'r') as f:
            matrix = np.loadtxt(f, delimiter=' ')
        matrix = matrix[:, 0:21]
        s = ''
        for hour in range(0,16):
            s += ','.join(map(str, matrix[hour]))
            if hour < 15:
                s += ' '
            elif hour == 15:
                s += '\n'
        f_out.write(s)
        
    f_out.close()

                
        
        
