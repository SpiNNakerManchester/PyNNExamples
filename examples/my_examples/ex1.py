import spynnaker8 as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt


simtime = 1000

# Write a network with a 1.0ms time step,
sim.setup(timestep=1.0)


# ...two current-based LIF neurons with default parameters
pop_1 = sim.Population(2, sim.IF_curr_exp(), label="pop_1")
# Consisting of two input source neurons ...with default parameters Have the
# first input neuron spike at time 0.0ms and the second spike at time 1.0ms.
input = sim.Population(2, sim.SpikeSourceArray(spike_times=[[0], [1], [50], [100], [200], [500]]),
                       label="input")
input_proj = sim.Projection(input, pop_1, sim.OneToOneConnector(),
                            synapse_type=sim.StaticSynapse(weight=5, delay=2))

# Record and plot the spikes received against time.
pop_1.record(["spikes"])
sim.run(simtime)

neo = pop_1.get_data(variables=["spikes"])
spikes = neo.segments[0].spiketrains
print(spikes)
sim.end()

plot.Figure(
    # plot spikes (or in this case spike)
    plot.Panel(spikes, yticks=True, xticks=True, markersize=5,
               xlim=(0, simtime)),
    title="Simple Example",
    annotations="Simulated with {}".format(sim.name())
)
plt.show()