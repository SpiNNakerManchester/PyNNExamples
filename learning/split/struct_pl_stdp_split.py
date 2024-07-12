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

import matplotlib.pyplot as plt
import numpy
import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
from spynnaker.pyNN.extra_algorithms.splitter_components import (
    SplitterAbstractPopulationVertexNeuronsSynapses, SplitterPoissonDelegate)

n_neurons = 192
simtime = 5000

sim.setup(timestep=1.0)

sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 64)
pre_splitter = SplitterAbstractPopulationVertexNeuronsSynapses(1, 128, False)
pre_pop = sim.Population(
    n_neurons, sim.IF_curr_exp(), label="Pre", additional_parameters={
        "splitter": pre_splitter})
post_splitter = SplitterAbstractPopulationVertexNeuronsSynapses(1, 128, False)
post_pop = sim.Population(
    n_neurons, sim.IF_curr_exp(), label="Post", additional_parameters={
        "splitter": post_splitter})
pre_noise = sim.Population(
    n_neurons, sim.SpikeSourcePoisson(rate=10.0), label="Noise_Pre",
    additional_parameters={"splitter": SplitterPoissonDelegate()})
post_noise = sim.Population(
    n_neurons, sim.SpikeSourcePoisson(rate=10.0), label="Noise_Post",
    additional_parameters={"splitter": SplitterPoissonDelegate()})

pre_pop.record("spikes")
post_pop.record("spikes")

training = sim.Population(
    n_neurons,
    sim.SpikeSourcePoisson(rate=10.0, start=1500.0, duration=1500.0),
    label="Training",
    additional_parameters={"splitter": SplitterPoissonDelegate()})

sim.Projection(pre_noise,  pre_pop,  sim.OneToOneConnector(),
               synapse_type=sim.StaticSynapse(weight=2.0))
sim.Projection(post_noise, post_pop, sim.OneToOneConnector(),
               synapse_type=sim.StaticSynapse(weight=2.0))

sim.Projection(training, pre_pop,  sim.OneToOneConnector(),
               synapse_type=sim.StaticSynapse(weight=5.0, delay=1.0))
sim.Projection(training, post_pop, sim.OneToOneConnector(),
               synapse_type=sim.StaticSynapse(weight=5.0, delay=10.0))

timing_rule = sim.SpikePairRule(tau_plus=20.0, tau_minus=20.0,
                                A_plus=0.5, A_minus=0.5)
weight_rule = sim.AdditiveWeightDependence(w_max=5.0, w_min=0.0)

partner_selection_last_neuron = sim.RandomSelection()
formation_distance = sim.DistanceDependentFormation(
    grid=[numpy.sqrt(n_neurons), numpy.sqrt(n_neurons)],
    sigma_form_forward=.5  # spread of feed-forward connections
)
elimination_weight = sim.RandomByWeightElimination(
    prob_elim_potentiated=0,  # no eliminations for potentiated synapses
    prob_elim_depressed=0,  # no elimination for depressed synapses
    threshold=0.5  # Use same weight as initial weight for static connections
)
structure_model_with_stdp = sim.StructuralMechanismSTDP(
    # Partner selection, formation and elimination rules from above
    partner_selection_last_neuron, formation_distance, elimination_weight,
    # Use this weight when creating a new synapse
    initial_weight=0,
    # Use this weight for synapses at start of simulation
    weight=0,
    # Use this delay when creating a new synapse
    initial_delay=5,
    # Use this weight for synapses at the start of simulation
    delay=5,
    # Maximum allowed fan-in per target-layer neuron
    s_max=64,
    # Frequency of rewiring in Hz
    f_rew=10 ** 4,
    # STDP rules
    timing_dependence=sim.SpikePairRule(
        tau_plus=20., tau_minus=20.0, A_plus=0.5, A_minus=0.5),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=5.)
)

plastic_projection = sim.Projection(
    pre_pop, post_pop,
    sim.FixedProbabilityConnector(0.),  # No initial connections
    synapse_type=structure_model_with_stdp,
    label="structurally_plastic_projection"
)

sim.run(simtime)

pre_neo = pre_pop.get_data(variables=["spikes"])
pre_spikes = pre_neo.segments[0].spiketrains

post_neo = post_pop.get_data(variables=["spikes"])
post_spikes = post_neo.segments[0].spiketrains

print(plastic_projection.get('weight', format="list"))

sim.end()

line_properties = [{'color': 'red', 'markersize': 5},
                   {'color': 'blue', 'markersize': 2}]

plot.Figure(
    # plot spikes
    plot.Panel(pre_spikes, post_spikes, yticks=True, xlim=(0, simtime),
               line_properties=line_properties),
    title="STDP Network Example",
    annotations=f"Simulated with {sim.name()}"
)
plt.show()
