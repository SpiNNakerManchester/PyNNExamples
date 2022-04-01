# Copyright (c) 2017-2020 The University of Manchester
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

import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
import pyNN.spiNNaker as sim

n_neurons = 100
simtime = 5000

sim.setup(timestep=1.0)

pre_pop = sim.Population(n_neurons, sim.IF_curr_exp(), label="Pre")
post_pop = sim.Population(n_neurons, sim.IF_curr_exp(), label="Post")
pre_noise = sim.Population(
    n_neurons, sim.SpikeSourcePoisson(rate=10.0), label="Noise_Pre")
post_noise = sim.Population(
    n_neurons, sim.SpikeSourcePoisson(rate=10.0), label="Noise_Post")

pre_pop.record("spikes")
post_pop.record("spikes")

training = sim.Population(
    n_neurons,
    sim.SpikeSourcePoisson(rate=10.0, start=1500.0, duration=1500.0),
    label="Training")

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

stdp_model = sim.STDPMechanism(timing_dependence=timing_rule,
                               weight_dependence=weight_rule,
                               weight=0.0, delay=5.0)

stdp_projection = sim.Projection(pre_pop, post_pop, sim.OneToOneConnector(),
                                 synapse_type=stdp_model)

sim.run(simtime)

pre_neo = pre_pop.get_data(variables=["spikes"])
pre_spikes = pre_neo.segments[0].spiketrains

post_neo = post_pop.get_data(variables=["spikes"])
post_spikes = post_neo.segments[0].spiketrains

print(stdp_projection.getWeights())

sim.end()

line_properties = [{'color': 'red', 'markersize': 5},
                   {'color': 'blue', 'markersize': 2}]

plot.Figure(
    # plot spikes
    plot.Panel(pre_spikes, post_spikes, yticks=True, xlim=(0, simtime),
               line_properties=line_properties),
    title="STDP Network Example",
    annotations="Simulated with {}".format(sim.name())
)
plt.show()
