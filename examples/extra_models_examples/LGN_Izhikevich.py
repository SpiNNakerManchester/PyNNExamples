# Copyright (c) 2020-2022 The University of Manchester
#
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.


"""
Modifications: Ported the model to Izhikevich's Conductance neurons
Developed as a part of the undergraduate project by Ishita Mediratta under
the guidance of Dr. Basabdatta Sen-Bhattacharya

The model is given Poisson input in the beta range with parameters tuned in
a way that they addhere to the plausible irregularity and synchrony values

Original Implementation:

Version uploaded on ModelDB October 2017.
Author:
Basabdatta Sen Bhattacharya, APT group, School of Computer Science,
University of Manchester, 2017.

If you are using the code,
please cite the original work on the model - details are:

B. Sen-Bhattacharya, T. Serrano-Gotarredona, L. Balassa, A. Bhattacharya,
A.B. Stokes, A. Rowley, I. Sugiarto, S.B. Furber,
"A spiking neural network model of the Lateral Geniculate Nucleus on the
SpiNNaker machine", Frontiers in Neuroscience, vol. 11 (454), 2017.

Free online access:
http://journal.frontiersin.org/article/10.3389/fnins.2017.00454/abstract
"""
# pylint: disable=pointless-string-statement

import pyNN.spiNNaker as p
import numpy as np
import math
from pyNN.random import RandomDistribution, NumpyRNG

# for plotting
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


def get_mean_rate(numCells, population):
    firing_rate = []      # format = < neuron_id, rate (spikes/ms) >

    for index in range(0, numCells):
        rate = len(population.segments[0].spiketrains[index])/TotalDuration
        firing_rate.append(rate)

    return sum(firing_rate)/len(firing_rate)


def calc_irregularity(segment):
    irregularity = 0
    isi_array = []
    for i in range(len(segment.spiketrains)):
        if(len(segment.spiketrains[i]) > 2):
            isi_array.append([])
            for j in range(len(segment.spiketrains[i])-1):
                isi_array[-1].append(
                    segment.spiketrains[i][j+1]-segment.spiketrains[i][j])
    for i in range(len(isi_array)):
        mean = np.mean(isi_array[i])
        sd = np.std(isi_array[i])
        cv = sd / mean
        irregularity += cv
    irregularity = irregularity / len(segment.spiketrains)
    return irregularity


def print_irregularity():
    print("TCR irregularity: ", calc_irregularity(TCR_spikes.segments[0]))
    print("IN irregularity: ", calc_irregularity(IN_spikes.segments[0]))
    print("TRN irregularity: ", calc_irregularity(TRN_spikes.segments[0]))


def calc_synchrony(segment):
    spike_counts = np.zeros(int(TotalDuration/2.0), dtype=int)
    for i in range(len(segment.spiketrains)):
        for j in range(len(segment.spiketrains[i])):
            index = math.floor(segment.spiketrains[i][j] / 2.0)
            spike_counts[index] += 1
    mean = np.mean(spike_counts)
    var = np.std(spike_counts) * np.std(spike_counts)
    synchrony = var / mean
    return synchrony


def print_synchrony():
    print("TCR synchrony: ", calc_synchrony(TCR_spikes.segments[0]))
    print("IN synchrony: ", calc_synchrony(IN_spikes.segments[0]))
    print("TRN synchrony: ", calc_synchrony(TRN_spikes.segments[0]))


""" Initialising Time and Frequency parameters """

# total duration of simulation
TotalDuration = int(1000)

# this is in ms.
Duration_Inp = int(1000)

# 50 ms at both start and end are disregarded to avoid transients
Start_Inp = int(0)
End_Inp = int(Start_Inp + Duration_Inp)

Rate_Inp = int(22)
Inp_isi = int(1000 / Rate_Inp)

""" Initialising Model connectivity parameters """

intra_pop_delay = RandomDistribution('uniform', (1, 3),
                                     rng=NumpyRNG(seed=85520))
intra_nucleus_delay = RandomDistribution('uniform', (1, 3),
                                         rng=NumpyRNG(seed=85521))
inter_nucleus_delay = RandomDistribution('uniform', (1, 3),
                                         rng=NumpyRNG(seed=85522))
inter_pop_delay = RandomDistribution('uniform', (1, 3),
                                     rng=NumpyRNG(seed=85523))
input_delay = inter_pop_delay

# # input_delay is the delay of the spike source hitting the neuronal pops
# # inter_pop_delay is the delay of spike communication between the different
# # populations of the model

# probabilities
p_trn2trn = 0.15
p_in2tcr = 0.1545  # 0.232
p_in2in = 0.236
p_tcr2trn = 0.35
p_trn2tcr = 0.1545  # 0.07
p_ret2tcr = 0.07
p_ret2in = 0.47

# weights
w_trn2trn = 1.0  # 0.06 # 1
w_in2tcr = 0.1  # 2
w_in2in = 0.35  # 2
w_tcr2trn = 0.115  # 0.01 # 2 # 0.2 and 0.3 give best value -> 0.25
w_trn2tcr = 0.03  # 2
w_ret2tcr = 0.275  # 0.35 # 0.1 # 1
w_ret2in = 0.275  # 0.35 # 0.1 # 1

""" Initialising Izhikevich spiking neuron model parameters.
We have used the conductance-based model here. """

# Tonic mode parameters
tcr_a_tonic = 0.02
tcr_b_tonic = 0.2
tcr_c_tonic = -65.0
tcr_d_tonic = 6.0
tcr_v_init_tonic = RandomDistribution('uniform', (-63.0, -67.0),
                                      rng=NumpyRNG(seed=85520))  # -65.0

in_a_tonic = 0.1
in_b_tonic = 0.2
in_c_tonic = -65.0
in_d_tonic = 6.0
in_v_init_tonic = RandomDistribution('uniform', (-68.0, -72.0),
                                     rng=NumpyRNG(seed=85521))  # -70.0

trn_a_tonic = 0.02
trn_b_tonic = 0.2
trn_c_tonic = -65.0
trn_d_tonic = 6.0
trn_v_init_tonic = RandomDistribution('uniform', (-73.0, -77.0),
                                      rng=NumpyRNG(seed=85522))  # -75.0

tcr_a = tcr_a_tonic
tcr_b = tcr_b_tonic
tcr_c = tcr_c_tonic
tcr_d = tcr_d_tonic
tcr_v_init = tcr_v_init_tonic

in_a = in_a_tonic
in_b = in_b_tonic
in_c = in_c_tonic
in_d = in_d_tonic
in_v_init = in_v_init_tonic

trn_a = trn_a_tonic
trn_b = trn_b_tonic
trn_c = trn_c_tonic
trn_d = trn_d_tonic
trn_v_init = trn_v_init_tonic

# tcr_b * tcr_v_init
tcr_u_init = RandomDistribution('uniform', (-15.0, -11.0),
                                rng=NumpyRNG(seed=85522))  # -13.0
# in_b * in_v_init
in_u_init = RandomDistribution('uniform', (-16.0, -12.0),
                               rng=NumpyRNG(seed=85522))  # -14.0
# trn_b * trn_v_init
trn_u_init = RandomDistribution('uniform', (-17.0, -13.0),
                                rng=NumpyRNG(seed=85522))  # -15.0

# a constant DC bias current; this is used here for testing the RS and FS
# characteristics of IZK neurons
current_Pulse = RandomDistribution('poisson', lambda_=3.0,
                                   rng=NumpyRNG(seed=85524))  # 5

# excitatory input time constant
tau_ex = 6.0

# inhibitory input time constant
tau_inh = 4.0

# reversal potentials
e_rev_ex = 0.0
e_rev_inh = -80.0

""" Starting the SpiNNaker Simulator """
p.setup(timestep=0.1)
# set number of neurons per core to 50, for the spike source to avoid clogging
# p.set_number_of_neurons_per_core(p.SpikeSourceArray, 50)

""" Defining each cell type as dictionary """

# THALAMOCORTICAL RELAY CELLS (TCR)
TCR_cell_params = {'a': tcr_a_tonic, 'b': tcr_b, 'c': tcr_c, 'd': tcr_d,
                   'tau_syn_E': tau_ex, 'tau_syn_I': tau_inh,
                   'i_offset': current_Pulse, 'e_rev_E': e_rev_ex,
                   'e_rev_I': e_rev_inh
                   }

TCR_initial_values = {'v': tcr_v_init, 'u': tcr_u_init}

# THALAMIC INTERNEURONS (IN)
IN_cell_params = {'a': in_a, 'b': in_b, 'c': in_c, 'd': in_d,
                  'tau_syn_E': tau_ex, 'tau_syn_I': tau_inh,
                  'i_offset': current_Pulse, 'e_rev_E': e_rev_ex,
                  'e_rev_I': e_rev_inh
                  }

IN_initial_values = {'v': in_v_init, 'u': in_u_init}

# THALAMIC RETICULAR NUCLEUS (TRN)
TRN_cell_params = {'a': trn_a, 'b': trn_b, 'c': trn_c, 'd': trn_d,
                   'tau_syn_E': tau_ex, 'tau_syn_I': tau_inh,
                   'i_offset': current_Pulse, 'e_rev_E': e_rev_ex,
                   'e_rev_I': e_rev_inh
                   }

TRN_initial_values = {'v': trn_v_init, 'u': trn_u_init}

""" Creating populations of each cell type """
scale_fact = 10
NumCellsTCR = 8*scale_fact
NumCellsIN = 2*scale_fact
NumCellsTRN = 4*scale_fact
TCR_pop = p.Population(
    NumCellsTCR, p.extra_models.Izhikevich_cond, TCR_cell_params,
    label='TCR_pop', initial_values=TCR_initial_values)
IN_pop = p.Population(
    NumCellsIN, p.extra_models.Izhikevich_cond, IN_cell_params,
    label='IN_pop', initial_values=IN_initial_values)
TRN_pop = p.Population(
    NumCellsTRN, p.extra_models.Izhikevich_cond, TRN_cell_params,
    label='TRN_pop', initial_values=TRN_initial_values)

""" Poisson input for TCR """
spike_source_TCR = p.Population(
    NumCellsTCR, p.SpikeSourcePoisson(rate=10, start=Start_Inp,
                                      duration=Duration_Inp),
    label='spike_source_TCR')

""" Poisson input for IN """
spike_source_IN = p.Population(
    NumCellsIN, p.SpikeSourcePoisson(rate=10, start=Start_Inp,
                                     duration=Duration_Inp),
    label='spike_source_IN')

""" Poisson Source to TCR population projections """
Proj0 = p.Projection(
    spike_source_TCR, TCR_pop, p.OneToOneConnector(),
    p.StaticSynapse(weight=w_ret2tcr, delay=input_delay),
    receptor_type='excitatory')


""" Poisson Source2IN """
Proj1 = p.Projection(
    spike_source_IN, IN_pop, p.OneToOneConnector(),
    p.StaticSynapse(weight=w_ret2in, delay=input_delay),
    receptor_type='excitatory')


""" TCR2TRN """
Proj2 = p.Projection(
    TCR_pop, TRN_pop, p.FixedProbabilityConnector(p_connect=p_tcr2trn),
    p.StaticSynapse(weight=w_tcr2trn, delay=inter_nucleus_delay),
    receptor_type='excitatory')


""" TRN2TCR """
Proj3 = p.Projection(
    TRN_pop, TCR_pop, p.FixedProbabilityConnector(p_connect=p_trn2tcr),
    p.StaticSynapse(weight=w_trn2tcr, delay=inter_nucleus_delay),
    receptor_type='inhibitory')


""" TRN2TRN """
Proj4 = p.Projection(
    TRN_pop, TRN_pop, p.FixedProbabilityConnector(p_connect=p_trn2trn),
    p.StaticSynapse(weight=w_trn2trn, delay=intra_pop_delay),
    receptor_type='inhibitory')


""" IN2TCR """
Proj5 = p.Projection(
    IN_pop, TCR_pop, p.FixedProbabilityConnector(p_connect=p_in2tcr),
    p.StaticSynapse(weight=w_in2tcr, delay=intra_nucleus_delay),
    receptor_type='inhibitory')


""" IN2IN """
Proj6 = p.Projection(
    IN_pop, IN_pop, p.FixedProbabilityConnector(p_connect=p_in2in),
    p.StaticSynapse(weight=w_in2in, delay=intra_pop_delay),
    receptor_type='inhibitory')

""" Recording simulation data"""

# recording the spikes and voltage
spike_source_TCR.record("spikes")
spike_source_IN.record("spikes")
# spike_source_periodic_TCR.record("spikes")
# spike_source_periodic_IN.record("spikes")
TCR_pop.record(("spikes", "v", "gsyn_exc", "gsyn_inh"))
IN_pop.record(("spikes", "v", "gsyn_exc", "gsyn_inh"))
TRN_pop.record(("spikes", "v", "gsyn_exc", "gsyn_inh"))

p.run(TotalDuration)

""" On simulation completion, extract the data off the spinnaker machine
memory """

# extracting the spike time data
# spikesourcepattern_TCR = spike_source_periodic_TCR.get_data("spikes")
# spikesourcepattern_IN = spike_source_periodic_IN.get_data("spikes")
spikesourcepattern_TCR = spike_source_TCR.get_data("spikes")
spikesourcepattern_IN = spike_source_IN.get_data("spikes")
TCR_spikes = TCR_pop.get_data("spikes")
IN_spikes = IN_pop.get_data("spikes")
TRN_spikes = TRN_pop.get_data("spikes")

# extracting the membrane potential data (in millivolts)
TCR_membrane_volt = TCR_pop.get_data("v")
IN_membrane_volt = IN_pop.get_data("v")
TRN_membrane_volt = TRN_pop.get_data("v")

# print TCR_membrane_volt.segments[0].analogsignals
TCR_gsyn_e = TCR_pop.get_data("gsyn_exc")
IN_gsyn_e = IN_pop.get_data("gsyn_exc")
TRN_gsyn_e = TRN_pop.get_data("gsyn_exc")

TCR_gsyn_i = TCR_pop.get_data("gsyn_inh")
IN_gsyn_i = IN_pop.get_data("gsyn_inh")
TRN_gsyn_i = TRN_pop.get_data("gsyn_inh")

print_irregularity()
print_synchrony()
print(get_mean_rate(NumCellsTCR, TCR_spikes)*1000)
print(get_mean_rate(NumCellsIN, IN_spikes)*1000)
print(get_mean_rate(NumCellsTRN, TRN_spikes)*1000)

""" Plotting """

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(TCR_spikes.segments[0].spiketrains, xlabel="Time/ms",
          xticks=True, ylabel="TCR Spikes Plots for TotalDuration",
          yticks=True, markersize=0.5, xlim=(1, TotalDuration), color='red'),
    Panel(IN_spikes.segments[0].spiketrains, xlabel="Time/ms",
          xticks=True, ylabel="IN Spikes Plots for TotalDuration",
          yticks=True, markersize=0.5, xlim=(1, TotalDuration), color='red'),
    Panel(TRN_spikes.segments[0].spiketrains, xlabel="Time/ms",
          xticks=True, ylabel="TRN Spikes Plots for TotalDuration",
          yticks=True, markersize=0.5, xlim=(1, TotalDuration), color='red'),
    Panel(TCR_membrane_volt.segments[0].filter(name="v")[0], xlabel="Time/ms",
          xticks=True, ylabel="TCR membrane voltage",
          yticks=True, markersize=0.5, xlim=(100, 400), legend=False),
    Panel(IN_membrane_volt.segments[0].filter(name="v")[0], xlabel="Time/ms",
          xticks=True, ylabel="IN membrane voltage",
          yticks=True, markersize=0.5, xlim=(100, 400), legend=False),
    Panel(TRN_membrane_volt.segments[0].filter(name="v")[0], xlabel="Time/ms",
          xticks=True, ylabel="TRN membrane voltage",
          yticks=True, markersize=0.5, xlim=(100, 400), legend=False),
    title="Effect of I_DC on periodic input, with Izhikevich_cond neurons",
    annotations="Simulated with {}".format(p.name())
)
# plt.savefig("Effect of I_DC on periodic input.png")
plt.show()

p.end()
