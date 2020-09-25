import spynnaker8 as pynn
import numpy as np
import matplotlib.pyplot as plt
from PyNN8Examples.eprop_testing.frozen_poisson import build_input_spike_train, frozen_poisson_variable_hz
from pyNN.random import NumpyRNG, RandomDistribution
from pyNN.utility.plotting import Figure, Panel
from spynnaker.pyNN.spynnaker_external_device_plugin_manager import SpynnakerExternalDevicePluginManager
from PyNN8Examples.eprop_testing.plot_graph import draw_graph_from_list, plot_learning_curve
from PyNN8Examples.eprop_testing.create_pops_for_incremental_learning import first_create_pops, next_create_pops, run_until
from PyNN8Examples.eprop_testing.incremental_config import *

np.random.seed(272727)

experiment_label, runtime, pynn, in_proj, recurrent_proj, out_proj, input_pop, neuron, readout_pop, \
from_list_in, from_list_rec, from_list_out = first_create_pops()

for i in range(1, 9, 2):
    new_connections_in, new_connections_rec, new_connections_out, good_performance = \
        run_until(experiment_label, runtime, pynn, in_proj, recurrent_proj, out_proj, input_pop, neuron, readout_pop,
                  from_list_in, from_list_rec, from_list_out)

    readout_neuron_params["number_of_cues"] += 2
    cycle_time = (number_of_cues * 150) + 1000 + 150

    experiment_label, runtime, pynn, in_proj, recurrent_proj, out_proj, input_pop, neuron, readout_pop, \
    from_list_in, from_list_rec, from_list_out = \
        next_create_pops(new_connections_in, new_connections_rec, new_connections_out)