# Copyright (c) 2017 The University of Manchester
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import matplotlib.pyplot as plt
from pyNN.random import RandomDistribution
import pyNN.utility.plotting as plot
import pyNN.spiNNaker as sim

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
