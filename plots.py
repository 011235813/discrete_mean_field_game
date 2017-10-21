import numpy as np
from numpy.linalg import norm
from scipy.stats import entropy
from scipy.stats import gaussian_kde
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from pylab import flipud
import pandas as pd


#plt.rcParams.update({'font.size': 12})
def plot_barchart(filename='chart_mfg_t8p06_s0p16_alpha12000_var_lag13_rnn.pdf'):
    N = 2
    mfg = (0.00396, 0.00581) # previous: (0.0028, 0.0045)
    mfg_std = (0.00352, 0.00184) # previous: (0.0014, 0.0018)
    
    ind = np.arange(N)  # the x locations for the groups
    width = 0.15       # the width of the bars
    
    fig, ax = plt.subplots()
    rects1 = ax.bar(ind, mfg, width, color='b', yerr=mfg_std)
    
    ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))
    
    var = (0.0166, 0.0178) # previous: (0.00359, 0.00456)
    var_std = (0.00526, 0.00400) # previous: (0.00101, 0.00104)
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
    pp = PdfPages('plots_irl/'+filename)
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


def JSD(P, Q):
    """
    Arguments:
    P,Q - discrete probability distribution
    
    Return:
    Jensen-Shannon divergence
    """

    # Replace all zeros by 1e-100
    P[P==0] = 1e-100
    Q[Q==0] = 1e-100

    P_normed = P / norm(P, ord=1)
    Q_normed = Q / norm(Q, ord=1)
    M = 0.5 * (P + Q)

    return 0.5 * (entropy(P,M) + entropy(Q,M))


def test_histogram(normed=False):
    mu, sigma = 100, 15
    x = mu + sigma * np.random.randn(10000)
    # x = np.ones(100)
    mu, sigma = 85, 15
    y = mu + sigma * np.random.randn(10000)
    # y = np.ones(100) * 2
    mu, sigma = 100, 15
    z = mu + sigma * np.random.randn(10000)
    # z = np.ones(100) * -1

    fig = plt.figure(1)
    plt.subplot(311)
    n_x, bins, patches = plt.hist(x, 50, normed=normed, facecolor='g', alpha=0.75)
    # y_min, y_max = axes.get_ylim()
    # print(y_min, y_max)
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])
    # plt.axis([40, 160, y_min, y_max])

    plt.subplot(312)
    n_y, bins, patches = plt.hist(y, 50, normed=normed, facecolor='r', alpha=0.75)
    axes = plt.gca()
    y_min, y_max = axes.get_ylim()
    print(y_min, y_max)
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=85,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])
    # plt.axis([40, 160, y_min, y_max])

    plt.subplot(313)
    n_z, bins, patches = plt.hist(z, 50, normed=normed, facecolor='b', alpha=0.75)
    axes = plt.gca()
    y_min, y_max = axes.get_ylim()
    print(y_min, y_max)
    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=115,\ \sigma=15$')
    # plt.axis([40, 160, 0, 0.03])        
    # plt.axis([40, 160, y_min, y_max])

    axes = plt.gca()
    # for item in ([axes.title, axes.xaxis.label, axes.yaxis.label] + axes.get_xticklabels()):
    #     item.set_fontsize(14)
    axes.xaxis.label.set_fontsize(14)
    plt.tight_layout()
    print(JSD(n_x, n_y), JSD(n_x,n_z))

    pp = PdfPages("test_histogram.pdf")
    pp.savefig(fig)
    pp.close()


def test_histogram_smooth(normed=True):
    mu, sigma = 100, 15
    data_1 = mu + sigma * np.random.randn(10000)
    mu, sigma = 85, 15
    data_2= mu + sigma * np.random.randn(10000)
    mu, sigma = 115, 15
    data_3 = mu + sigma * np.random.randn(100)

    fig = plt.figure()

    plt.xlabel('Smarts')
    plt.ylabel('Probability')
    plt.title('Histogram of IQ')
    plt.text(60, .025, r'$\mu=100,\ \sigma=15$')

    density1 = gaussian_kde(data_1)
    xs = np.linspace(50, 150,200)
    # density1.covariance_factor = lambda : .25
    # density1._compute_covariance()
    plt.plot(xs, density1(xs))

    density2 = gaussian_kde(data_2)
    xs = np.linspace(50, 150,200)
    # density2.covariance_factor = lambda : .25
    # density2._compute_covariance()
    plt.plot(xs, density2(xs))

    density3 = gaussian_kde(data_3)
    xs = np.linspace(50, 150,200)
    # density3.covariance_factor = lambda : .25
    # density3._compute_covariance()
    plt.plot(xs, density3(xs))    

    pp = PdfPages("test_histogram_smooth.pdf")
    pp.savefig(fig)
    pp.close()


def test_heatmap():
    # Generate some data that where each slice has a different range
    # (The overall range is from 0 to 2)
    arr1 = np.random.random((15,15))
    arr2 = np.random.random((15,15))
    data = np.stack([arr1, arr2], axis=0)
    
    # Plot each slice as an independent subplot
    fig, axes = plt.subplots(nrows=1, ncols=2)
    # for dat, ax in zip(data, axes.flat):
    #     # The vmin and vmax arguments specify the color limits
    #     im = ax.imshow(dat, cmap='hot', vmin=0, vmax=1)
    ax1 = axes[0]
    im = ax1.imshow(arr1, cmap='hot', vmin=0, vmax=1)
    ax1.set_title('hello')
    major_ticks = np.arange(0, 15, 3)                                         
    ax1.set_xticks(major_ticks)
    ax1.set_yticks(major_ticks)

    ax2 = axes[1]
    im = ax2.imshow(arr2, cmap='hot', vmin=0, vmax=1)
    ax2.set_title('bye')
    major_ticks = np.arange(0, 15, 3)                                         
    ax2.set_xticks(major_ticks)           
    ax2.set_yticks(major_ticks)

    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.3, 0.05, 0.4])
    fig.colorbar(im, cax=cbar_ax)
    
    # Make an axis for the colorbar on the right side
    # cax = fig.add_axes([0.9, 0.1, 0.03, 0.8]) # xmin, ymin, dx, dy
    # fig.colorbar(im, cax=cax)
    
    # plt.tight_layout()
    pp = PdfPages("test_heatmap.pdf")
    pp.savefig(fig, bbox_inches='tight')
    pp.close()


def test_colorbar():
    # Builtin colourmap "seismic" has the blue-white-red
    #   scale you want
    a = np.zeros((16,16), dtype=np.float32)
    a[0,0] = 1.0
    b = np.ones((16,16), dtype=np.float32) * 0.5
    b[0,0] = 1.0
    
    fig = plt.figure()
    plt.subplot(121, aspect='equal')
    # plt.pcolor(data, cmap=plt.cm.seismic, vmin=0, vmax=2)
    plt.pcolor(flipud(a), cmap='hot', vmin=0, vmax=1)
    plt.colorbar()

    plt.subplot(122, aspect='equal')
    plt.pcolor(flipud(b), cmap='hot', vmin=0, vmax=1)
    plt.colorbar()

    pp = PdfPages("test_colorbar.pdf")
    pp.savefig(fig)
    pp.close()


def test_heatmap_simple():
    data = np.array([[1,2],[3,4]])
    
    # Plot each slice as an independent subplot
    fig = plt.figure()
    ax = plt.gca()

    im = ax.imshow(data, cmap='hot', vmin=0, vmax=4)
    ax.set_title('hello')
    major_ticks = np.arange(0, 2, 1) 
    ax.set_xticks(major_ticks)
    ax.set_yticks(major_ticks)
    ax.set_xlabel('x label')
    ax.set_ylabel('y label')

    fig.colorbar(im)

    # plt.tight_layout()
    pp = PdfPages("test_heatmap_simple.pdf")
    pp.savefig(fig, bbox_inches='tight')
    pp.close()    
