import numpy as np
import matplotlib.pylab as plt
from signal_prep import *

results_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition' \
                    '/HTM/2e2i'

htm_file = np.load(results_directory + '/spike_trains.npz')
weights_file = np.load(results_directory + '/varying_weights.npz')

test_id = 222

g_syn = htm_file['g_syn']
cell_voltage_plot_8(g_syn, plt, 6000., [],id=test_id,title="g_syn col {}".format(test_id),
                        )

mem_v = htm_file['mem_v']
cell_voltage_plot_8(mem_v, plt, 6000., [],id=test_id,title="col {}".format(test_id),
                        )

varying_weights = weights_file['varying_weights_cd'][1]

id_incoming = [(pre,weight) for (pre, post, weight) in varying_weights if post==test_id]

plt.figure()
plt.plot(id_incoming)

plt.show()