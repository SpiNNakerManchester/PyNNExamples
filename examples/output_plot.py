from signal_prep import *
import numpy as np
import matplotlib.pylab as plt

input_directory = "/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains"
# input_file = np.load(input_directory+"/IC_spikes/brainstem_asc_des_60s_60dB.npz")
input_file = np.load(input_directory+"/IC_spikes/brainstem_asc_des_a_i_u_60s_20dB.npz")

input_spikes = input_file['ic_times']
onset_times = input_file['onset_times']
onset_times_s = []
for times in onset_times:
    onset_times_s.append([time/1000. for time in times])

test_file = np.load(input_directory  + "/spatial_pooler_mult.npz")
column_spikes = test_file['column_spikes']
inh_spikes = test_file['inh_pop_spikes']
varying_weights = test_file['varying_weights']
varying_weights_inh= test_file['varying_weights_inh']

max_time = 0
for neuron in np.asarray(column_spikes):
    if neuron.size > 0 and neuron.max() > max_time:
        max_time = neuron.max().item()

# weight_dist_plot(varying_weights, 1, plt, 0.0,6./8, title="input->column weight distribution")
# connection_hist_plot(varying_weights, pre_size=1000, post_size=1000, plt=plt,
#                      title="input->column")  # ,weight_min=av_weight)

# weight_dist_plot(varying_weights_inh, 1, plt, 0.0, 10./8, title="inh->column weight distribution")
# connection_hist_plot(varying_weights_inh, pre_size=1000, post_size=1000, plt=plt,
#                      title="inh->column")  # ,weight_min=av_weight)
#
# spike_raster_plot_8(column_spikes,plt,max_time/1000.,len(column_spikes)+1,0.001,title="output pop activity",
#                     onset_times=onset_times_s,pattern_duration=100.)
#
# spike_raster_plot_8(input_spikes,plt,max_time/1000.,len(column_spikes)+1,0.001,title="input pop activity",
#                     onset_times=onset_times_s,pattern_duration=100.)
#
# spike_raster_plot_8(inh_spikes,plt,max_time/1000.,len(column_spikes)+1,0.001,title="inh pop activity",
#                     onset_times=onset_times_s,pattern_duration=100.)

# sparsity_matrix = sparsity_measure(onset_times,column_spikes,onset_window=100.,from_time=0.)
# plt.figure('output pop sparsity')
# for stimulus in sparsity_matrix:
#     plt.plot(stimulus)
#
sparsity_matrix = sparsity_measure(onset_times,input_spikes,onset_window=100.,from_time=0.)
plt.figure('input pop sparsity')
for stimulus in sparsity_matrix:
    plt.plot(stimulus)
#
# sparsity_matrix = sparsity_measure(onset_times,inh_spikes,onset_window=100.,from_time=0.)
# plt.figure('inh pop sparsity')
# for stimulus in sparsity_matrix:
#     plt.plot(stimulus)

plt.show()