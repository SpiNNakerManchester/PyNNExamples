import numpy as np

#read final spikes file
results_directory = "/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/HTM/4_patterns_2sequences_50columns_16active_neurons_2.0Hz_800cds_16.0Taup_30.0taumin_0.2alpha_spike_pair_structural_plasticty/"
final_spikes = np.load(results_directory+"/final_spike_trains.npz")

ms_onset_times = final_spikes['ms_onset_times']
active_spikes = final_spikes['final_active_spike_train']
cd_spikes = final_spikes['final_cd_spike_train']

#ABCD XBCY test
B_given_A = ms_onset_times[1][-1]
B_given_X = ms_onset_times[5][-1]

C_given_B_given_A = ms_onset_times[2][-1]
C_given_B_given_X = ms_onset_times[6][-1]

context_onsets = [B_given_A,B_given_X,C_given_B_given_A,C_given_B_given_X]
number_of_representations_to_test = 4
representation_neurons = [[]for _ in range(number_of_representations_to_test)]

time_window=5.

B_given_A_neurons=[]
for neuron_id,times in enumerate(active_spikes):
    for time in times:
        if time:
            for index,context in enumerate(context_onsets):
                if time>=context and time<(context+time_window):
                    representation_neurons[index].append(neuron_id)

import csv
from itertools import izip_longest
with open(results_directory+"/ABCDXBCY_neurons.csv","w+") as f:
    writer = csv.writer(f)
    for values in izip_longest(*representation_neurons):
        writer.writerow(values)

print representation_neurons