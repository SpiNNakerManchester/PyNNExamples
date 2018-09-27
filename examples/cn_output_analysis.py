from signal_prep import *
import numpy as np
import matplotlib.pylab as plt
from pyNN.random import NumpyRNG, RandomDistribution

# diagonal_width = 5.#2.#26.#
# diagonal_sparseness = 1.
# in2out_sparse = .67 * .67 / diagonal_sparseness
# dist = max(int(359 / 1000.), 1)
# sigma = dist * diagonal_width
# conn_num = int(sigma / in2out_sparse)
# connections = normal_dist_connection_builder(1000, 359, RandomDistribution, NumpyRNG(), conn_num, dist, sigma)
# max_post = 0
# max_pre = 0
# for (pre,post) in connections:
#     if post > max_post:
#         max_post = post
#     if pre > max_pre:
#         max_pre = pre
input_dir = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
cn_data = np.load(input_dir + "/cn_13.5_1kHz_50dB_0.1ms_timestep_1s.npz")

an_spikes = cn_data['an_spikes']
ch_spikes = cn_data['ch_spikes']
on_spikes = cn_data['on_spikes']
onset_times = cn_data['onset_times']

spikes = an_spikes[959]#ch_spikes[959]
isi = [(spikes[i] - spikes[i-1])/0.4 for i,_ in enumerate(spikes) if i>0]
# plt.figure()
# plt.hist(isi,bins=1000)
# plt.show()

# spike_raster_plot_8(an_spikes,plt,1.,len(an_spikes)+1,title="an pop activity")
# spike_raster_plot_8(on_spikes,plt,1.,len(on_spikes)+1,title="on pop activity")
# spike_raster_plot_8(ch_spikes,plt,1.,len(ch_spikes)+1,title="ch pop activity")

an_psth_spikes = repeat_test_spikes_gen(an_spikes,500,onset_times)
ch_psth_spikes = repeat_test_spikes_gen(ch_spikes,163,onset_times)
on_psth_spikes = repeat_test_spikes_gen(on_spikes,140,onset_times)

psth_index = 1

# spike_raster_plot_8(ch_psth_spikes[psth_index],plt,0.2,250.+1,title="ch repeat pop activity")
# spike_raster_plot_8(on_psth_spikes[psth_index],plt,0.2,250.+1,title="on repeat pop activity")


# psth_plot_8(plt,numpy.arange(len(an_psth_spikes[0])),an_psth_spikes[0],bin_width=0.001,duration=0.2,title="PSTH_AN")
# psth_plot_8(plt,numpy.arange(len(ch_psth_spikes[psth_index])),ch_psth_spikes[psth_index],bin_width=0.0001,duration=0.2,title="PSTH_CH")
# psth_plot_8(plt,numpy.arange(len(on_psth_spikes[psth_index])),on_psth_spikes[psth_index],bin_width=0.0001,duration=0.2,title="PSTH_ON")

psth_plot_8(plt,numpy.arange(150,200),ch_spikes,bin_width=0.0001,duration=1.,title="PSTH_CH_fibres")


plt.show()