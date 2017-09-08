import spynnaker8 as sim

sim.setup(timestep=1.0)
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)

pop_1 = sim.Population(size=1, cellclass=sim.IF_curr_exp(), label="pop_1")
input = sim.Population(size=1, cellclass=sim.SpikeSourceArray(
    spike_times=[0]), label="input")
input_proj = sim.Projection(
    presynaptic_population=input,
    postsynaptic_population=pop_1,
    connector=sim.OneToOneConnector(),
    synapse_type = sim.StaticSynapse(weight=5, delay=1))
pop_1.record(variables=["spikes","v"])
runtime = 10
sim.run(simtime=runtime)

neo = pop_1.get_data(variables=["spikes","v"])
spikes = neo.segments[0].spiketrains
print spikes
v = neo.segments[0].filter(name='v')
print v
sim.end()

import matplotlib.pyplot as plt
import pyNN.utility.plotting as plot
plot.Figure(
    # plot spikes (or in this case spike)
    plot.Panel(spikes, yticks=True, markersize=5, xlim=(0, runtime)),
    # plot voltage for first ([0]) neuron
    plot.Panel(v[0], ylabel="Membrane potential (mV)",
               data_labels=[pop_1.label], yticks=True, xlim=(0, runtime)),
    title="Simple Example",
    annotations="Simulated with {}".format(sim.name())
)
plt.show()
