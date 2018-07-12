''' A balanced network simulation'''

import pyNN.spiNNaker as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
from pyNN.random import RandomDistribution
from neo.io import NeoHdf5IO

sim.setup(timestep=0.1)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp,80)

number_neurons = 256

random_weights_ex = RandomDistribution('normal_clipped', mu=0.1, sigma=0.1, low=0, high=10)
random_delays_ex = RandomDistribution('normal_clipped', mu=1.5, sigma=0.75, low=0, high=1)

random_weights_inh = RandomDistribution('normal_clipped', mu=-0.4, sigma=0.1, high=0, low=-10)
random_delays_inh = RandomDistribution('normal_clipped', mu=0.75, sigma=0.375, low=0, high=1)

ex_pop = sim.Population(int(number_neurons*0.8), sim.IF_curr_exp(), label="ex_pop")
inh_pop = sim.Population(int(number_neurons*0.2), sim.IF_curr_exp(), label="inh_pop" )

ex_poisson = sim.Population(int(number_neurons*0.8), sim.SpikeSourcePoisson(rate=1000))
inh_poisson = sim.Population(int(number_neurons*0.2), sim.SpikeSourcePoisson(rate=1000))

ex_input_proj = sim.Projection(ex_poisson, ex_pop, sim.OneToOneConnector(), sim.StaticSynapse(weight=0.1, delay=0.1))
inh_input_proj = sim.Projection(inh_poisson, inh_pop, sim.OneToOneConnector(), synapse_type=sim.StaticSynapse(weight=0.1, delay=0.1))

ex_inh_proj = sim.Projection(ex_pop, inh_pop, sim.FixedProbabilityConnector(0.1), sim.StaticSynapse(random_weights_ex, random_delays_ex))
ex_ex_proj = sim.Projection(ex_pop, ex_pop, sim.FixedProbabilityConnector(0.1), sim.StaticSynapse(random_weights_ex, random_delays_ex))

inh_ex_proj = sim.Projection(inh_pop, ex_pop, sim.FixedProbabilityConnector(0.1), sim.StaticSynapse(random_weights_inh, random_delays_inh))
inh_inh_proj = sim.Projection(inh_pop, inh_pop, sim.FixedProbabilityConnector(0.1), sim.StaticSynapse(random_weights_inh, random_delays_inh))

random_initial = RandomDistribution('uniform', (-65, -55))

ex_pop.initialize(v=random_initial)
inh_pop.initialize(v=random_initial)


inh_pop.record(('spikes'))
ex_pop.record(["spikes"])
ex_poisson.record(["spikes"])



#inh_data = NeoHdf5IO("inhdata")
#ex_data = NeoHdf5IO("exdata")

#inh_pop.write_data(inh_data)
#ex_pop.write_data(ex_data)

simtime=1000
sim.run(simtime)

neo = ex_pop.get_data(variables=["spikes"])
spikes = neo.segments[0].spiketrains

sim.end()

plot.Figure(
# plot voltage for first ([0]) neuron
#plot.Panel(v, ylabel="Membrane potential (mV)",
#data_labels=[pop_1.label], yticks=True, xlim=(0, simtime)),
# plot spikes (or in this case spike)
plot.Panel(spikes, yticks=True, markersize=5, xlim=(0, simtime)),
title="Simple Example",
annotations="Simulated with {}".format(sim.name())
)
plt.show()

