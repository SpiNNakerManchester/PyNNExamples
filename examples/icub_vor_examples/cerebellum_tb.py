# Copyright (c) 2021 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import pyNN.spiNNaker as sim
import numpy as np
import matplotlib.pyplot as plt

from pyNN.utility.plotting import Figure, Panel
from pyNN.random import RandomDistribution

L_RATE = 2
H_RATE = 20

# cerebellum test bench
runtime = 1000

# Synapsis parameters
gc_pc_weights = 0.005
mf_vn_weights = 0.0005
pc_vn_weights = 0.01
cf_pc_weights = 0.0
mf_gc_weights = 0.5
go_gc_weights = 0.002
input_weights = 0.0025
mf_go_weights = 0.1

# Network parameters
num_MF_neurons = 100
num_GC_neurons = 2000
num_GOC_neurons = 100
num_PC_neurons = 200
num_VN_neurons = 200
num_CF_neurons = 200

# Random distribution for synapses delays and weights (MF and GO)
delay_distr = RandomDistribution('uniform', (1.0, 10.0))
weight_distr_MF = RandomDistribution(
    'uniform', (mf_gc_weights*0.8, mf_gc_weights*1.2))
seed_MF = 85524
weight_distr_GO = RandomDistribution(
    'uniform', (go_gc_weights*0.8, go_gc_weights*1.2))
seed_GO = 24568


# Post-synapse population
neuron_params = {
    "v_thresh": -50,
    "v_reset": -70,
    "v_rest": -65,
    "i_offset": 0  # DC input
}

# Learning parameters cos rule (MF to VN)
min_weight_c = 0
max_weight_c = 0.005
pot_alpha_c = 0.001  # this is alpha in the paper
beta_c = 11
sigma_c = 201
initial_weight_c = 0.001  # max_weight_c # 0.0005
# initial_weight_c = 0.05
plastic_delay_c = 4

# Learning parameters sin rule (GrC to PC)
min_weight_s = 0
max_weight_s = 0.01
pot_alpha_s = 0.01
t_peak_s = 100
initial_weight_s = max_weight_s  # 0.0001
plastic_delay_s = 4
weight_dist_pfpc = RandomDistribution('uniform',
                                      (initial_weight_s*0.8,
                                       initial_weight_s*1.2))
seed_PFPC = 24534

sim.setup(timestep=1.)

# Sensorial Activity: input activity from vestibulus (will come from the head
# IMU, now it is a test bench)
# We simulate the output of the head encoders with a sinusoidal function. Each
# "sensorial activity" value is derived from the head position and velocity.
# From that value, we generate the mean firing rate of the MF neurons (later
# this will be an input that will come from the robot, through the spinnLink)
# the neurons that are active depend on the value of the sensorial activity.
# For each a gaussian is created centered on a specific neuron.

# Prepare variables once at beginning
MAX_RATE = 600.0
MAX_AMPLITUDE = 0.8
RELATIVE_AMPLITUDE = 1.0
_head_pos = []
_head_vel = []

i = np.arange(0, 1000, 0.001)
for t in i:
    desired_speed = -np.cos(
        t * 2 * np.pi) * MAX_AMPLITUDE * RELATIVE_AMPLITUDE * 2.0 * np.pi
    desired_pos = -np.sin(
        t * 2 * np.pi) * MAX_AMPLITUDE * RELATIVE_AMPLITUDE
    _head_pos.append(desired_pos)
    _head_vel.append(desired_speed)


def sensorial_activity(pt):
    # pt is a single time point at which we measure the head encoder's output
    # single point over time
    head_pos = _head_pos[pt]
    head_vel = _head_vel[pt]

    head_pos = ((head_pos + 0.8) / 1.6)
    head_vel = ((head_vel + 0.8 * 2 * np.pi) / (1.6 * 2 * np.pi))

    if head_pos > 1.0:
        head_pos = 1.0
    elif head_pos < 0.0:
        head_pos = 0.0
    if head_vel > 1.0:
        head_vel = 1.0
    elif head_vel < 0.0:
        head_vel = 0.0

    min_rate = 0.0
    max_rate = MAX_RATE
    sigma = 0.02
    MF_pos_activity = np.zeros((50))
    MF_vel_activity = np.zeros((50))

    # generate gaussian distributions around the neuron tuned to a given
    # sensorial activity
    for i in range(50):
        mean = float(i) / 50.0 + 0.01
        gaussian = np.exp(
            -((head_pos - mean) * (head_pos - mean))/(2.0 * sigma * sigma))
        MF_pos_activity[i] = min_rate + gaussian * (max_rate - min_rate)

    for i in range(50):
        mean = float(i) / 50.0 + 0.01
        gaussian = np.exp(
            -((head_vel - mean) * (head_vel - mean))/(2.0 * sigma * sigma))
        MF_vel_activity[i] = min_rate + gaussian * (max_rate - min_rate)

    sa_mean_freq = np.concatenate((MF_pos_activity, MF_vel_activity))
    out = [sa_mean_freq, head_pos, head_vel]
    return out


# Error Activity: error from eye and head encoders
def error_activity(error_):

    IO_agonist = np.zeros((100))
    IO_antagonist = np.zeros((100))

    IO_agonist[:] = H_RATE
    IO_antagonist[:] = L_RATE

    ea_rate = np.concatenate((IO_agonist, IO_antagonist))

    return ea_rate


def process_VN_spiketrains(VN_spikes, t_start):
    total_spikes = 0
    for spiketrain in VN_spikes.segments[0].spiketrains:
        s = spiketrain.as_array()[
            np.where(spiketrain.as_array() >= t_start)[0]]
        total_spikes += len(s)

    return total_spikes


###############################################################
# Create Populations
###############################################################

# Create MF population - fake input population that will be substituted by
# external input from robot

MF_population = sim.Population(
    num_MF_neurons,  # number of sources
    sim.SpikeSourcePoisson,  # source type
    # {'rate': sa_mean_freq},  # source spike times
    {'rate': sensorial_activity(0)[0]},  # source spike times
    label="MFLayer",  # identifier
    additional_parameters={"max_rate": MAX_RATE, "seed": seed_MF}
    )

# Create GOC population
GOC_population = sim.Population(num_GOC_neurons, sim.IF_cond_exp(),
                                label='GOCLayer',
                                additional_parameters={"seed": seed_GO})

# create PC population
PC_population = sim.Population(
    num_PC_neurons,  # number of neurons
    sim.IF_cond_exp(**neuron_params),  # Neuron model
    label="Purkinje Cell",  # identifier
    additional_parameters={"seed": seed_PFPC}
    )

# create VN population
VN_population = sim.Population(
    num_VN_neurons,  # number of neurons
    sim.IF_cond_exp(**neuron_params),  # Neuron model
    label="Vestibular Nuclei",  # identifier
    additional_parameters={"seed": seed_PFPC}
    )

# Create GrC population
GC_population = sim.Population(num_GC_neurons, sim.IF_curr_exp(),
                               label='GCLayer',
                               additional_parameters={"seed": seed_GO})

# generate fake error (it should be calculated from sensorial activity in error
# activity, but we skip it and just generate an error from -1.5 to 1.5)
err = -0.7  # other values to test: -0.3 0 0.3 0.7

# Create CF population - fake input population that will be substituted by
# external input from robot
CF_population = sim.Population(
    num_CF_neurons,  # number of sources
    sim.SpikeSourcePoisson,  # source type
    # {'rate': sa_mean_freq},  # source spike times
    {'rate': error_activity(err)},  # source spike times
    label="CFLayer",  # identifier
    additional_parameters={"max_rate": MAX_RATE}
    )

###############################################################
# Create connections
###############################################################

# Create MF-GO connections
mf_go_connections = sim.Projection(MF_population,
                                   GOC_population,
                                   sim.OneToOneConnector(),
                                   sim.StaticSynapse(delay=delay_distr,
                                                     weight=mf_go_weights),
                                   receptor_type='excitatory')

# Create MF-GC and GO-GC connections
float_num_MF_neurons = float(num_MF_neurons)

list_GOC_GC = []
list_MF_GC = []
list_GOC_GC_2 = []
# projections to subpopulations
# https://github.com/SpiNNakerManchester/sPyNNaker8/issues/168)
for i in range(num_MF_neurons):
    GC_medium_index = int(
        round((i / float_num_MF_neurons) * num_GC_neurons))
    GC_lower_index = GC_medium_index - 40
    GC_upper_index = GC_medium_index + 60

    if (GC_lower_index < 0):
        GC_lower_index = 0

    elif (GC_upper_index > num_GC_neurons):
        GC_upper_index = num_GC_neurons

    for j in range(GC_medium_index - GC_lower_index):
        list_GOC_GC.append(
            (i, GC_lower_index + j,  # go_gc_weights, 1)
             weight_distr_GO.next(), delay_distr.next()))

    for j in range(GC_medium_index + 20 - GC_medium_index):
        list_MF_GC.append(
            (i, GC_medium_index + j,  # mf_gc_weights, 1)
             weight_distr_MF.next(), delay_distr.next()))

    for j in range(GC_upper_index - GC_medium_index - 20):
        list_GOC_GC_2.append(
            (i, GC_medium_index + 20 + j,  # go_gc_weights, 1)
             weight_distr_GO.next(), delay_distr.next()))

GO_GC_con1 = sim.Projection(
    GOC_population, GC_population, sim.FromListConnector(list_GOC_GC),
    receptor_type='inhibitory')  # this should be inhibitory

MF_GC_con2 = sim.Projection(
    MF_population, GC_population, sim.FromListConnector(list_MF_GC),
    receptor_type='excitatory')

GO_GC_con3 = sim.Projection(
    GOC_population, GC_population, sim.FromListConnector(list_GOC_GC_2),
    receptor_type='inhibitory')


# Create PC-VN connections
pc_vn_connections = sim.Projection(
    PC_population, VN_population, sim.OneToOneConnector(),
    # receptor_type='GABA', # Should these be inhibitory?
    synapse_type=sim.StaticSynapse(delay=delay_distr, weight=pc_vn_weights),
    receptor_type='inhibitory')

# Create MF-VN learning rule - cos
mfvn_plas = sim.STDPMechanism(
    timing_dependence=sim.extra_models.TimingDependenceMFVN(beta=beta_c,
                                                            sigma=sigma_c),
    weight_dependence=sim.extra_models.WeightDependenceMFVN(
        w_min=min_weight_c, w_max=max_weight_c, pot_alpha=pot_alpha_c),
    weight=initial_weight_c, delay=delay_distr)

# Create MF to VN connections
mf_vn_connections = sim.Projection(
    MF_population, VN_population, sim.AllToAllConnector(),
    # Needs mapping as FromListConnector to make efficient
    synapse_type=mfvn_plas, receptor_type="excitatory")

# Create projection from PC to VN -- replaces "TEACHING SIGNAL"
pc_vn_connections = sim.Projection(
    PC_population, VN_population, sim.OneToOneConnector(),
    sim.StaticSynapse(weight=0.0, delay=1.0),
    receptor_type="excitatory")  # "TEACHING SIGNAL"

# create PF-PC learning rule - sin
pfpc_plas = sim.STDPMechanism(
    timing_dependence=sim.extra_models.TimingDependencePFPC(t_peak=t_peak_s),
    weight_dependence=sim.extra_models.WeightDependencePFPC(
        w_min=min_weight_s, w_max=max_weight_s, pot_alpha=pot_alpha_s),
    weight=initial_weight_s, delay=delay_distr
    )

# Create PF-PC connections
pf_pc_connections = sim.Projection(
    GC_population, PC_population, sim.AllToAllConnector(),
    synapse_type=pfpc_plas, receptor_type="excitatory")

# Create IO-PC connections. This synapse with "receptor_type=COMPLEX_SPIKE"
# propagates the learning signals that drive the plasticity mechanisms in
# GC-PC synapses
cf_pc_connections = sim.Projection(
    CF_population, PC_population, sim.OneToOneConnector(),
    # receptor_type='COMPLEX_SPIKE',
    synapse_type=sim.StaticSynapse(delay=1.0, weight=cf_pc_weights),
    receptor_type='excitatory')

MF_population.record(['spikes'])
CF_population.record(['spikes'])
GC_population.record('all')
GOC_population.record(['spikes'])
VN_population.record('all')  # VN_population.record(['spikes'])
PC_population.record(['spikes'])

samples_in_repeat = 99
sample_time = 10
repeats = 1
total_runtime = 0
VN_transfer_func = []

for i in range(samples_in_repeat):
    sim.run(sample_time)

    VN_spikes = VN_population.get_data('spikes')
    VN_transfer_func.append(process_VN_spiketrains(VN_spikes, total_runtime))

    total_runtime += sample_time

    print(total_runtime)

    MF_population.set(rate=sensorial_activity(total_runtime)[0])

total_runtime = runtime*repeats

MF_spikes = MF_population.get_data('spikes')
CF_spikes = CF_population.get_data('spikes')
GC_spikes = GC_population.get_data('all')
GOC_spikes = GOC_population.get_data('spikes')
VN_spikes = VN_population.get_data('all')  # VN_population.get_data('spikes')
PC_spikes = PC_population.get_data('spikes')

mfvn_weights = mf_vn_connections.get('weight', 'list', with_address=False)
pfpc_weights = pf_pc_connections.get('weight', 'list', with_address=False)

# Plot
F = Figure(
    Panel(MF_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, total_runtime),
          xlabel='MF_spikes'),
    Panel(CF_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, total_runtime),
          xlabel='CF_spikes'),
    Panel(GC_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, total_runtime),
          xlabel='GC_spikes'),
    Panel(GOC_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, total_runtime),
          xlabel='GOC_spikes'),
    Panel(PC_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, total_runtime),
          xlabel='PC_spikes'),
    Panel(VN_spikes.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, total_runtime),
          xlabel='VN_spikes'),
    Panel(VN_spikes.segments[0].filter(name='gsyn_inh')[0],
          ylabel="Membrane potential (mV)", yticks=True,
          xlim=(0, total_runtime))
    )
plt.show()

plt.figure()
plt.plot(mfvn_weights,
         label='mf-vn weights (init: {})'.format(initial_weight_c))
plt.legend()

plt.figure()
plt.plot(pfpc_weights, color='orange',
         label='pf-pc weights (init: {})'.format(initial_weight_s))
plt.legend()

print(VN_transfer_func)

plt.figure()
plt.plot(VN_transfer_func)

plt.show()

sim.end()
print("job done")
