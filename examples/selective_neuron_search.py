import numpy as np
from signal_prep import *
import matplotlib.pylab as plt
from scipy.io import loadmat

pattern_spikes = np.load('./pattern_spikes.npy')
#max_time = int(max(pattern_spikes)/1000.)
duration = 60000
num_pattern_firings = 10
#pattern_b_spikes = np.load('./pattern_b_spikes.npy')
#take final 10% of pattern spikes 60*10*0.9:60*10
stimulus_times=[]
for pattern in pattern_spikes:
    #stimulus_times.append(pattern[duration*num_pattern_firings*0.9:duration*num_pattern_firings-1].tolist())
    stimulus_times.append(pattern[pattern>duration*0.9])
#stimulus_times.append(pattern_b_spikes[60*10*0.9:60*10-1].tolist())

spike_train = np.load('./target_spikes.npy')

#TODO add pattern on period to plot
spike_raster_plot_8(spike_train, plt, duration/1000, 100 + 1, 0.001,
                    title="target pop activity")

plt.show()

max_id = len(spike_train)

time_window = 10.
counts,selective_neuron_ids,significant_spike_count = neuron_correlation(spike_train,time_window,stimulus_times,max_id)

print "significant spike count: {}".format(significant_spike_count)
import matplotlib.pyplot as plt
max_count = counts.max()
plt.figure()
title = "{}ms post-stimulus spike count for target layer".format(time_window)
plt.title(title)
plt.xlabel("neuron ID")
plt.ylabel("spike count")
plt.plot(counts.T)
legend_string=[]
for i in range(len(stimulus_times)):
    legend_string.append("stimulus {}".format(i+1))
plt.legend(legend_string)
plt.ylim((0,max_count+1))

for i in range(len(selective_neuron_ids)):
    print selective_neuron_ids[i]

plt.show()