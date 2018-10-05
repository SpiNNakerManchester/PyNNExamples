import numpy as np
from signal_prep import *
import matplotlib.pylab as plt
from scipy.io import loadmat

pattern_a_spikes = np.load('./pattern_a_spikes.npy')
#take final 10% of pattern spikes 60*10*0.9:60*10

stimulus_times = [54000,55000,56000,57000,58000,59000]

#spike_raster_plot(spike_train,plt=plt,duration=duration,ylim=1000 ,scale_factor=0.001,title='A1')
#spike_raster_plot(spikes_train_an_ms,plt=plt,duration=duration,ylim=1000 ,scale_factor=0.001,title='AN')
#psth_plot(plt,numpy.arange(1000),spike_train,bin_width=0.001,duration=duration,scale_factor=0.001,title="PSTH_A1_pre")
#plt.show()

#only interested in responses to final few stimuli to observe plasticity effects
stimulus_times_final = []
for stimulus_time in stimulus_times:
    stimulus_times_final.append(stimulus_time[-10:])

counts,selective_neuron_ids,significant_spike_count = neuron_correlation(spike_train,time_window,stimulus_times_final,max_id)
print "significant spike count: {}".format(significant_spike_count)
import matplotlib.pyplot as plt
max_count = counts.max()
plt.figure()
title = "{}ms post-stimulus spike count for AC A1 layer".format(time_window)
plt.title(title)
plt.xlabel("neuron ID")
plt.ylabel("spike count")
plt.plot(counts.T)
plt.legend(["stimulus 'one'", "stimulus 'two'"])
plt.ylim((0,max_count+1))

#plt.figure()
#plt.hist(counts[0])
#plt.figure()
#plt.hist(counts[1])

np.save('./selective_ids.npy',selective_neuron_ids)
for i in range(len(selective_neuron_ids)):
    print selective_neuron_ids[i]

plt.show()
