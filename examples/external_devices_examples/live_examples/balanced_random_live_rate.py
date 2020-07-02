# Copyright (c) 2017-2019 The University of Manchester
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
import pylab
import numpy
from pyNN.utility.plotting import Figure, Panel
import time

p.setup(timestep=0.1)
p.set_number_of_neurons_per_core(p.SpikeSourcePoisson, 50)
n_neurons = 500
n_exc = int(round(n_neurons * 0.8))
n_inh = int(round(n_neurons * 0.2))
weight_exc = 0.1
weight_inh = -5.0 * weight_exc
weight_input = 0.001

pop_input = p.Population(100, p.SpikeSourcePoisson(rate=0), label="Input")

pop_exc = p.Population(n_exc, p.IF_curr_exp, label="Excitatory",
                       additional_parameters={"spikes_per_second": 100})
pop_inh = p.Population(n_inh, p.IF_curr_exp, label="Inhibitory",
                       additional_parameters={"spikes_per_second": 100})
stim_exc = p.Population(
    n_exc, p.SpikeSourcePoisson(rate=1000.0), label="Stim_Exc")
stim_inh = p.Population(
    n_inh, p.SpikeSourcePoisson(rate=1000.0), label="Stim_Inh")

delays_exc = RandomDistribution(
    "normal_clipped", mu=1.5, sigma=0.75, low=1.0, high=14.4)
weights_exc = RandomDistribution(
    "normal_clipped", mu=weight_exc, sigma=0.1, low=0, high=numpy.inf)
conn_exc = p.FixedProbabilityConnector(0.1)
synapse_exc = p.StaticSynapse(weight=weights_exc, delay=delays_exc)
delays_inh = RandomDistribution(
    "normal_clipped", mu=0.75, sigma=0.375, low=1.0, high=14.4)
weights_inh = RandomDistribution(
    "normal_clipped", mu=weight_inh, sigma=0.1, low=-numpy.inf, high=0)
conn_inh = p.FixedProbabilityConnector(0.1)
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
    "normal_clipped", mu=1.5, sigma=0.75, low=1.0, high=14.4)
weights_input = RandomDistribution(
    "normal_clipped", mu=weight_input, sigma=0.01, low=0, high=numpy.inf)
p.Projection(pop_input, pop_exc, p.AllToAllConnector(), p.StaticSynapse(
    weight=weights_input, delay=delays_input))

pop_exc.initialize(v=RandomDistribution("uniform", low=-65.0, high=-55.0))
pop_inh.initialize(v=RandomDistribution("uniform", low=-65.0, high=-55.0))

pop_exc.record("spikes")

poisson_control = p.external_devices.SpynnakerPoissonControlConnection(
    poisson_labels=[pop_input.label], local_port=None)

p.external_devices.add_poisson_live_rate_control(
    pop_input, database_notify_port_num=poisson_control.local_port)


def start_callback(label, connection):
    for rate in [50, 10, 20]:
        time.sleep(10.0)
        connection.set_rates(label, [(i, rate) for i in range(100)])


print(pop_input.label)
poisson_control.add_start_resume_callback(pop_input.label, start_callback)

p.run(5000)

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
