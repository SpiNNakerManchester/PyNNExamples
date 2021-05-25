import spynnaker8 as sim
import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
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

stdp_model = sim.STDPMechanism(timing_dependence=timing_rule,
                               weight_dependence=weight_rule,
                               weight=0.0, delay=5.0)

stdp_projection = sim.Projection(pre_pop, post_pop, sim.OneToOneConnector(),
                                 # sim.StaticSynapse(0, 5.0))
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
