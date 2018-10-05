from signal_prep import connection_hist_plot
import numpy as np
import matplotlib.pylab as plt

results_directory = "/home/rjames/Dropbox (The University of Manchester)/EarProject/Pattern_recognition/HTM/4_patterns_1sequences_50columns_16active_neurons_2.0Hz_800cds_16.0Taup_30.0taumin_0.2alpha_spike_pair_structural_plasticty"


varying_weights = np.load(results_directory+"/varying_weights.npz")

varying_weights_test = varying_weights['varying_weights_cd']

column_size = 16#32
number_of_columns = 50
active_pop_size = column_size*number_of_columns
cd_pop_size = int(1 * active_pop_size)
connection_hist_plot(varying_weights_test,pre_size=active_pop_size,post_size=cd_pop_size,plt=plt)
print
plt.show()