from signal_prep import *
import numpy as np
import matplotlib.pylab as plt

# input_directory = "./examples"#""/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains"
input_directory = "/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains"
spatial_pooler_file = np.load(input_directory+'/spatial_pooler.npz')
sparse_spikes = spatial_pooler_file['column_spikes']
#sparse_spikes = spatial_pooler_file['inh_pop_spikes']
#sparse_spikes = np.load(input_directory+"/IC_spikes/ic_spikes_asc_test_60s.npy")
IC_spikes = np.load(input_directory +"/IC_spikes/ic_spikes_asc_test_60s.npy")

max_time = 0
for neuron in IC_spikes:
    if neuron.size>0 and neuron.max() > max_time:
        max_time = neuron.max().item()
spike_raster_plot_8(IC_spikes,plt,max_time/1000.,len(IC_spikes)+1,0.001,title="input activity")
plt.show()

number_of_inputs = 1000
jitter_input = 10.
isi = 100
n_repeats = 100
input_spikes = [[]for _ in range(number_of_inputs)]

stim_1_ids = np.random.choice(range(number_of_inputs),number_of_inputs*0.4,replace=False)
stim_2_ids = np.random.choice(range(number_of_inputs),number_of_inputs*0.2,replace=False)

onset_times = []
onset_times.append([i*isi - jitter_input/2. for i in range(n_repeats)])
onset_times.append([i*isi + isi/2. - jitter_input/2. for i in range(n_repeats)])

for i in range(n_repeats):

    for neuron in stim_1_ids:
        input_spikes[neuron].append(i * isi + int(jitter_input * (np.random.rand() - 0.5)))
    for neuron in stim_2_ids:
        input_spikes[neuron].append(i * isi + isi/2. + int(jitter_input * (np.random.rand() - 0.5)))

# sparsity_matrix = sparsity_measure(onset_times,input_spikes,onset_window=jitter_input,from_time=0.)
#
# plt.figure('synthetic input')
# for stimulus in sparsity_matrix:
#     plt.plot(stimulus)

sparse_size = len(sparse_spikes)

max_time = 0
for neuron in sparse_spikes:
    if neuron.size>0 and neuron.max() > max_time:
        max_time = neuron.max().item()


# ear_file = np.load("/home/rjames/SpiNNaker_devel/OME_SpiNN/spike_trains_asc_test_60s.npz")
ear_file = np.load("/home/rjames/SpiNNaker_devel/OME_SpiNN/spinnakear_asc_des_60s.npz")
onset_times = ear_file['onset_times']
onset_window = 100

sparsity_matrix = sparsity_measure(onset_times,IC_spikes,onset_window=onset_window,from_time=0.)

plt.figure('IC_spikes')
for stimulus in sparsity_matrix:
    plt.plot(stimulus)


onset_times = spatial_pooler_file['onset_times']
onset_window = spatial_pooler_file['onset_window']

onset_times_s = []
for times in onset_times:
    onset_times_s.append([time/1000. for time in times])

# spike_raster_plot_8(sparse_spikes, plt, max_time / 1000., sparse_size + 1, 0.001, title="output pop activity",
#                     onset_times=onset_times_s, pattern_duration=100.)
# plt.show()

mem_v = spatial_pooler_file['sparse_mem_v']#np.load(input_directory+"/IC_spikes/sparse_mem.npy")

# target_duration_ms = 10.*60.*1000.
# n_repeats = np.ceil(target_duration_ms / max_time)
# onset_times_long = []
# for stimulus in onset_times:
#     onset_times_long.append([j*max_time+time for j in xrange(int(n_repeats)) for time in stimulus])
# onset_times=onset_times_long

#take final 10% of times
final_times = 0#max_time * 0.8#

sparsity_matrix = sparsity_measure(onset_times,sparse_spikes,onset_window=onset_window,from_time=final_times)

plt.figure()
for stimulus in sparsity_matrix:
    plt.plot(stimulus)
    #plt.ylim((0,2.5))



# spike_raster_plot_8(sparse_spikes,plt,max_time/1000.,sparse_size+1,0.001,title="output pop activity",
#                     onset_times=onset_times_s,pattern_duration=100.)

# ids = None#502
# cell_voltage_plot_8(mem_v, plt, max_time, 1.,scale_factor=0.001,id=ids,title='output pop')


plt.show()