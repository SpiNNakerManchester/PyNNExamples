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

import spynnaker8 as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
from pyNN.random import RandomDistribution

n_neurons = 1000
n_exc = int(round(n_neurons * 0.8))
n_inh = int(round(n_neurons * 0.2))
simtime = 1000

sim.setup(timestep=0.1)

pop_exc = sim.Population(n_exc, sim.IF_curr_exp(), label="Excitatory")
pop_inh = sim.Population(n_inh, sim.IF_curr_exp(), label="Inhibitory")
stim_exc = sim.Population(n_exc, sim.SpikeSourcePoisson(rate=10.0),
                          label="Stim_Exc")
stim_inh = sim.Population(n_inh, sim.SpikeSourcePoisson(rate=10.0),
                          label="Stim_Inh")

synapse_exc = sim.StaticSynapse(weight=0.2, delay=2.0)
synapse_inh = sim.StaticSynapse(weight=-1.0, delay=2.0)
sim.Projection(pop_exc, pop_exc, sim.FixedProbabilityConnector(0.1),
               synapse_type=synapse_exc, receptor_type="excitatory")
sim.Projection(pop_exc, pop_inh, sim.FixedProbabilityConnector(0.1),
               synapse_type=synapse_exc, receptor_type="excitatory")
sim.Projection(pop_inh, pop_inh, sim.FixedProbabilityConnector(0.1),
               synapse_type=synapse_inh, receptor_type="inhibitory")
sim.Projection(pop_inh, pop_exc, sim.FixedProbabilityConnector(0.1),
               synapse_type=synapse_inh, receptor_type="inhibitory")

delays = RandomDistribution("uniform", parameters_pos=[1.0, 1.6])
conn_stim = sim.OneToOneConnector()
synapse_stim = sim.StaticSynapse(weight=2.0, delay=delays)

sim.Projection(stim_exc, pop_exc, conn_stim, synapse_type=synapse_stim,
               receptor_type="excitatory")
sim.Projection(stim_inh, pop_inh, conn_stim, synapse_type=synapse_stim,
               receptor_type="excitatory")

pop_exc.initialize(
    v=RandomDistribution("uniform", parameters_pos=[-65.0, -55.0]))
pop_inh.initialize(
    v=RandomDistribution("uniform", parameters_pos=[-65.0, -55.0]))
pop_exc.record("spikes")
sim.run(simtime)

neo = pop_exc.get_data(variables=["spikes"])
spikes = neo.segments[0].spiketrains

sim.end()

plot.Figure(
    # plot spikes
    plot.Panel(spikes, yticks=True, markersize=5, xlim=(0, simtime)),
    title="Balanced Random Network Example",
    annotations="Simulated with {}".format(sim.name())
)
plt.show()
