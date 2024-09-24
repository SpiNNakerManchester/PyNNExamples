# Copyright (c) 2017 The University of Manchester
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

# Demonstration of the WTA connector in use.  There are two populations both
# receiving input from the same Poisson source.  One population has a
# self-connection with a WTA connector, which will attempt to ensure that only
# one neuron in the population spikes at a time.  As neuron 2 has a higher rate
# of input than the others, it will be the "winner" more often than the others.

# Note that SpiNNaker does not send "instantaneous" spikes, so there can be
# times where two neurons spike in the same time step.

# The output graph shows the difference in the outputs of the two populations.

import pyNN.spiNNaker as sim
import matplotlib.pyplot as plt
import numpy

sim.setup(1.0)

pop = sim.Population(10, sim.IF_curr_exp(), label="pop")
wta = sim.Population(10, sim.IF_curr_exp(), label="wta")
stim = sim.Population(
    10, sim.SpikeSourcePoisson(
        rate=[10, 10, 20, 10, 10, 10, 10, 10, 10, 10]),
    label="stim")
pop.record("spikes")
wta.record("spikes")

sim.Projection(
    stim, pop, sim.OneToOneConnector(), sim.StaticSynapse(weight=5.0))
sim.Projection(
    stim, wta, sim.OneToOneConnector(), sim.StaticSynapse(weight=5.0))
sim.Projection(
    wta, wta, sim.extra_models.WTAConnector(), sim.StaticSynapse(weight=10.0),
    receptor_type="inhibitory")

sim.run(10000)

pop_spikes = pop.get_data("spikes").segments[0].spiketrains
wta_spikes = wta.get_data("spikes").segments[0].spiketrains

sim.end()

# Plot the spikes
for spiketrain in pop_spikes:
    y = numpy.ones_like(spiketrain) * spiketrain.annotations["source_index"]
    line, = plt.plot(spiketrain, y.magnitude * 2, "r|",
                     label="Without WTA")
for spiketrain in wta_spikes:
    y = numpy.ones_like(spiketrain) * spiketrain.annotations["source_index"]
    line_2, = plt.plot(spiketrain, (y.magnitude * 2) + 1, "b|",
                       label="With WTA")
plt.xlabel("Time (ms)")
plt.title("Simple example")
plt.legend(handles=[line, line_2], loc=9)
plt.ylim(-2, 24)
plt.yticks([], [])
plt.show()
