import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import pandas as pd

#plt.rcParams.update({'font.size': 12})
def plot_barchart():
    N = 2
    mfg = (0.0028, 0.0045) #(0.00267, 0.00429) #(0.0147, 0.0222)
    mfg_std = (0.0014, 0.0018) #(0.00162, 0.0019) #(0.009, 0.005)
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15       # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, mfg, width, color='b', yerr=mfg_std)
    
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    var = (0.00359, 0.00456) #(0.0271, 0.0292)
    var_std = (0.00101, 0.00104) #(0.0156, 0.0115)
    rects2 = ax.bar(ind + width, var, width, color='r', yerr=var_std)
    
    rnn = (0.613119, 0.594970)
    rnn_std = (0.007203, 0.008202)
    rects3 = ax.bar(ind + 2*width, rnn, width, color='m', yerr=rnn_std)
    
    start, end = ax.get_ylim()
    step = (end-start)/5
    ax.yaxis.set_ticks(np.arange(start, end+step*0.9, step))
    
    # add some text for labels, title and axes ticks
    ax.set_ylabel('JSD (log scale)')
    ax.set_title('Average test error over 10 days')
    ax.set_xticks(ind + width / 2)
    ax.set_xticklabels(('Final distribution', 'Entire trajectory (averaged)'))
    ax.set_yscale('log')
    
    for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                 ax.get_xticklabels()):
        item.set_fontsize(14)
    
    ax.legend((rects1[0], rects2[0], rects3[0]), ('MFG', 'VAR', 'RNN'), bbox_to_anchor=(0.65,1), prop={'size':14})
    
    #plt.show()
    pp = PdfPages('plots/chart_mfg_t8p86_s0p16_alpha12000_var_lag22_rnn.pdf')
    pp.savefig(fig)
    pp.close()


def plot_reward_convergence(filename='reward_convergence.csv', outfile='plots_irl/reward_convergence.pdf'):

    df = pd.read_table('results'+'/'+filename, delimiter=',', header=0)
    # indices = df.index.values
    indices = df['iteration']
    fig = plt.figure()
    plt.plot(indices, df['reward_demo_avg'], label='demostration')
    plt.plot(indices, df['reward_gen_avg'], label='generated')
    plt.ylabel('Avg reward over state-action pairs')
    plt.xlabel('Iterations')
    plt.legend(loc='best')
    plt.title("Average reward for demo and generated samples over iterations")
    pp = PdfPages(outfile)
    pp.savefig(fig)
    pp.close()


def test_histogram(normed=True):
    mu, sigma = 100, 15
    # x = mu + sigma * np.random.randn(10000)
    x = np.ones(100)
    mu, sigma = 85, 15
    # y = mu + sigma * np.random.randn(10000)
    y = np.ones(100) * 2
    mu, sigma = 115, 15
    # z = mu + sigma * np.random.randn(10000)
    z = np.ones(100) * -1

    fig = plt.figure(1)
    plt.subplot(311)
    n, bins, patches = plt.hist(x, 50, normed=normed, facecolor='g', alpha=0.75)
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])

    plt.subplot(312)
    n, bins, patches = plt.hist(y, 50, normed=normed, facecolor='r', alpha=0.75)
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=85,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])

    plt.subplot(313)
    n, bins, patches = plt.hist(z, 50, normed=normed, facecolor='b', alpha=0.75)
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=115,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])        

    pp = PdfPages("test_histogram.pdf")
    pp.savefig(fig)
    pp.close()

    # plt.grid(True)
    # plt.show()    
