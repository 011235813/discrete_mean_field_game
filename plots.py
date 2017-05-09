import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages


#plt.rcParams.update({'font.size': 12})

N = 2
mfg = (0.0147, 0.0222)
mfg_std = (0.009, 0.005)

ind = np.arange(N)  # the x locations for the groups
width = 0.25       # the width of the bars

fig, ax = plt.subplots()
rects1 = ax.bar(ind, mfg, width, color='b', yerr=mfg_std)

ax.ticklabel_format(style='sci', axis='y', scilimits=(0,0))

var = (0.0271, 0.0292)
var_std = (0.0156, 0.0115)
rects2 = ax.bar(ind + width, var, width, color='r', yerr=var_std)

start, end = ax.get_ylim()
step = (end-start)/5
ax.yaxis.set_ticks(np.arange(start, end+step, step))

# add some text for labels, title and axes ticks
ax.set_ylabel('L1 norm of difference')
ax.set_title('Test error averaged over 11 days')
ax.set_xticks(ind + width / 2)
ax.set_xticklabels(('Final distribution', 'Entire trajectory (averaged)'))

for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
             ax.get_xticklabels()):
    item.set_fontsize(14)

ax.legend((rects1[0], rects2[0]), ('MFG', 'VAR'), bbox_to_anchor=(0.7,1), prop={'size':14})

#plt.show()
pp = PdfPages('plots/perf_mfg_var_m5d9.pdf')
pp.savefig(fig)
pp.close()
