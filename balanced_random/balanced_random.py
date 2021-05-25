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

import spynnaker8 as p
from pyNN.random import RandomDistribution
import matplotlib.pyplot as pylab
import numpy
from pyNN.utility.plotting import Figure, Panel

p.setup(timestep=0.1)
p.set_number_of_neurons_per_core(p.IF_curr_exp, 64)
p.set_number_of_neurons_per_core(p.SpikeSourcePoisson, 64)
n_neurons = 500
n_exc = int(round(n_neurons * 0.8))
n_inh = int(round(n_neurons * 0.2))
weight_exc = 0.1
weight_inh = -5.0 * weight_exc
weight_input = 0.001
rng = p.NumpyRNG(0)

pop_input = p.Population(100, p.SpikeSourcePoisson(rate=0.0),
                         additional_parameters={
                             "max_rate": 50.0,
                             "seed": 0},
                         label="Input")

pop_exc = p.Population(n_exc, p.IF_curr_exp, label="Excitatory")
pop_inh = p.Population(n_inh, p.IF_curr_exp, label="Inhibitory")
stim_exc = p.Population(
    n_exc, p.SpikeSourcePoisson(rate=1000.0), label="Stim_Exc",
    additional_parameters={"seed": 1})
stim_inh = p.Population(
    n_inh, p.SpikeSourcePoisson(rate=1000.0), label="Stim_Inh",
    additional_parameters={"seed": 2})

delays_exc = RandomDistribution(
    "normal_clipped", mu=1.5, sigma=0.75, low=1.0, high=1.6, rng=rng)
weights_exc = RandomDistribution(
    "normal_clipped", mu=weight_exc, sigma=0.1, low=0, high=numpy.inf, rng=rng)
conn_exc = p.FixedProbabilityConnector(0.1, rng=rng)
synapse_exc = p.StaticSynapse(weight=weights_exc, delay=delays_exc)
delays_inh = RandomDistribution(
    "normal_clipped", mu=0.75, sigma=0.375, low=1.0, high=1.6, rng=rng)
weights_inh = RandomDistribution(
    "normal_clipped", mu=weight_inh, sigma=0.1, low=-numpy.inf, high=0,
    rng=rng)
conn_inh = p.FixedProbabilityConnector(0.1, rng=rng)
synapse_inh = p.StaticSynapse(weight=weights_inh, delay=delays_inh)
p.Projection(
    pop_exc, pop_exc, conn_exc, synapse_exc, receptor_type="excitatory")
p.Projection(
    pop_exc, pop_inh, conn_exc, synapse_exc, receptor_type="excitatory")
p.Projection(
    pop_inh, pop_inh, conn_inh, synapse_inh, receptor_type="inhibitory")
p.Projection(
    pop_inh, pop_exc, conn_inh, synapse_inh, receptor_type="inhibitory")

conn_stim = p.OneToOneConnector()
synapse_stim = p.StaticSynapse(weight=weight_exc, delay=1.0)
p.Projection(
    stim_exc, pop_exc, conn_stim, synapse_stim, receptor_type="excitatory")
p.Projection(
    stim_inh, pop_inh, conn_stim, synapse_stim, receptor_type="excitatory")

delays_input = RandomDistribution(
    "normal_clipped", mu=1.5, sigma=0.75, low=1.0, high=1.6, rng=rng)
weights_input = RandomDistribution(
    "normal_clipped", mu=weight_input, sigma=0.01, low=0, high=numpy.inf,
    rng=rng)
p.Projection(pop_input, pop_exc, p.AllToAllConnector(), p.StaticSynapse(
    weight=weights_input, delay=delays_input))

pop_exc.initialize(
    v=RandomDistribution("uniform", low=-65.0, high=-55.0, rng=rng))
pop_inh.initialize(
    v=RandomDistribution("uniform", low=-65.0, high=-55.0, rng=rng))

pop_exc.record("spikes")

p.run(1000)

pop_input.set(rate=50.0)
p.run(1000)

pop_input.set(rate=10.0)
p.run(1000)

pop_input.set(rate=20.0)
p.run(1000)

data = pop_exc.get_data("spikes")
end_time = p.get_current_time()

p.end()

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(data.segments[0].spiketrains,
          yticks=True, markersize=2.0, xlim=(0, end_time)),
    title="Balanced Random Network",
    annotations="Simulated with {}".format(p.name())
)
pylab.show()
