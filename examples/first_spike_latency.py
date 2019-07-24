import numpy as np
import matplotlib.pyplot as plt
from signal_prep import spike_raster_plot_8
from scipy.io import loadmat

# open the sgn spikes
input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'
file_name = '/home/rjames/Dropbox (The University of Manchester)/EarProject/MAP_BS_17/results/n_channels_experiment/fsl.mat'
spinnakear =False
dBs = [0]#[i*10 for i in range(10)]
fsl = []
for test_index,dB in enumerate(dBs):
    if spinnakear:
        sim_file = np.load(input_directory+'/ear_tone_1000Hz_stereo_200an_fibres_{}dB_0s.npz'.format(dB))
        stimulus = sim_file['stimulus']
        # sg_data = sim_file['sg_data']
        sg_data = sim_file['ear_data']
        # vrr = sg_data[0]['debug']
        Fs = sim_file['Fs']
    else:
        matlab_file = loadmat(file_name)
        an_spikes = matlab_file['ANoutput']
        sg_data = [an_spikes]
        stimulus = matlab_file['stimulus']
        Fs=50e3
    fsl.append([])
    for ear_index,sg in enumerate(sg_data):
        # get hold of the stimulus onset time
        latencies = []
        spike_times = []
        for i,sample in enumerate(stimulus[ear_index]):
            if sample > 4e-6:
                onset_time = int(i * (1./Fs)*1000.) #ms
                break
        if spinnakear:
            # target_neurons = sg.segments[0].spiketrains[1::2]
            target_neurons = sg['spikes'][1::2]
            # target_neurons = sg['spikes'][0::2]
        else:
            # target_neurons = an_spikes[:len(an_spikes)/2]
            target_neurons = an_spikes[len(an_spikes)/2:]
        for neuron in target_neurons:
            if spinnakear:
                for time in neuron:
                    if time.item() > onset_time:
                        latencies.append(time.item() - onset_time)
                        break
            else:
                times = np.nonzero(neuron)[0] * (1000. / Fs)
                spike_times.append(times)
                for time in times:
                    if time.item()>onset_time:
                        latencies.append(time.item()-onset_time)
                        break

        fsl[test_index].append(np.mean(latencies))
a = np.mean(fsl)
if spinnakear:
    spike_raster_plot_8(target_neurons, plt, 0.35, len(target_neurons) + 1, 0.001, title="sg pop activity {}".format(test_index),markersize=1)
else:
    spike_raster_plot_8(spike_times, plt, 0.35, len(spike_times) + 1, 0.001, title="sg pop activity {}".format(test_index),markersize=1)
plt.figure()
plt.hist(latencies,bins=100)

plt.figure()
plt.plot(stimulus[0])

plt.show()


