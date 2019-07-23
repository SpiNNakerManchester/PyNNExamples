import numpy as np
import matplotlib.pyplot as plt
from signal_prep import spike_raster_plot_8

# open the sgn spikes
input_directory = '/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/spike_trains/IC_spikes'

dBs = [0,100]#[i*10 for i in range(10)]
fsl = []
for test_index,dB in enumerate(dBs):
    sim_file = np.load(input_directory+'/ear_tone_1000Hz_stereo_1000an_fibres_{}dB_0s.npz'.format(dB))
    stimulus = sim_file['stimulus']
    sg_data = sim_file['sg_data']
    Fs = sim_file['Fs']
    fsl.append([])
    for ear_index,sg in enumerate(sg_data):
        # get hold of the stimulus onset time
        latencies = []
        for i,sample in enumerate(stimulus[ear_index]):
            if sample > 4e-6:
                onset_time = int(i * (1./Fs)*1000.) #ms
        target_neurons = sg.segments[0].spiketrains[::2]
        for neuron in target_neurons:
            for time in neuron:
                if time.item()>onset_time:
                    latencies.append(time.item()-onset_time)
                    break
        fsl[test_index].append(np.mean(latencies))
a = np.mean(fsl)
print
# find the spike latency for each neuron type

# calculate the


