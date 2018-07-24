from signal_prep import sparsity_measure,spike_raster_plot_8
import numpy as np
import matplotlib.pylab as plt

input_directory = "/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains"
sparse_spikes = np.load(input_directory+'/IC_spikes/sparse_spikes.npy')
sparse_size = len(sparse_spikes)

max_time = 0
for neuron in sparse_spikes:
    if neuron.size>0 and neuron.max() > max_time:
        max_time = neuron.max().item()

ear_file = np.load("/home/rjames/SpiNNaker_devel/OME_SpiNN/spike_trains_asc_test_60s.npz")
onset_times = ear_file['onset_times']

target_duration_ms = 10.*60.*1000.
n_repeats = np.ceil(target_duration_ms / max_time)
onset_times_long = []
for stimulus in onset_times:
    onset_times_long.append([j*max_time+time for j in xrange(int(n_repeats)) for time in stimulus])

#take final 10% of times
final_times = 0#max_time * 0.8

sparsity_matrix = sparsity_measure(onset_times_long,sparse_spikes,onset_window=100.,from_time=final_times)

plt.figure()
for stimulus in sparsity_matrix:
    plt.plot(stimulus)

onset_times_s = []
for times in onset_times:
    onset_times_s.append([time/1000. for time in times])
spike_raster_plot_8(sparse_spikes,plt,max_time/1000.,sparse_size+1,0.001,title="output pop activity",onset_times=onset_times_s,pattern_duration=100.)


plt.show()