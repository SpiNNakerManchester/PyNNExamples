
import numpy as np
import matplotlib.pylab as plt
from scipy.io import loadmat

spike_trains = np.load('./pre_pop_spikes.npy')
#spike_trains = np.load('../../Brainstem/ac_spikes.npy')
spike_trains_list=[]
for neuron in spike_trains:
    for spike in neuron:
        spike_trains_list.append(spike.item())

# matlab_spikes = loadmat('../../Brainstem/ic_spikes.mat')
# ids = [id for id in matlab_spikes['output'][0]]
# spike_trains_list = [time*1000. for time in matlab_spikes['output'][1]]

max_time = int(max(spike_trains_list))
bin_width_ms = 1.
num_bins = int(np.ceil(float(max_time)/bin_width_ms))

sync_count =np.zeros(num_bins)
for bin in range(num_bins):
    bin_index = int(bin*bin_width_ms)
    for time in range(bin_index,int(bin_index+bin_width_ms)):
        sync_count[bin]+=spike_trains_list.count(time)

print "average number of synchronised neurons per timestep = {}".format(np.mean(sync_count))
plt.figure()
plt.title("measured synchronised neurons over {}ms time bins".format(bin_width_ms))
x = np.linspace(0,max_time,num_bins)
plt.plot(x,sync_count)
plt.ylabel("number of synchronised neurons")
plt.xlabel("time (ms)")
plt.show()


