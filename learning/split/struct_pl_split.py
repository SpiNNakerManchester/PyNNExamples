import matplotlib.pyplot as plt
import numpy
import pyNN.utility.plotting as plot
import pyNN.spiNNaker as sim
from spynnaker.pyNN.extra_algorithms.splitter_components import (
    SplitterAbstractPopulationVertexNeuronsSynapses, SplitterPoissonDelegate)

n_neurons = 64
simtime = 5000

sim.setup(timestep=1.0)

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

# Structurally plastic connection between pre_pop and post_pop
partner_selection_last_neuron = sim.RandomSelection()
formation_distance = sim.DistanceDependentFormation(
    grid=[numpy.sqrt(n_neurons), numpy.sqrt(n_neurons)],
    sigma_form_forward=.5  # spread of feed-forward connections
)
elimination_weight = sim.RandomByWeightElimination(
    threshold=.2  # Use same weight as initial weight for static connections
)
structure_model_without_stdp = sim.StructuralMechanismStatic(
    # Partner selection, formation and elimination rules from above
    partner_selection_last_neuron, formation_distance, elimination_weight,
    # Use this weight when creating a new synapse
    initial_weight=5,
    # Use this weight for synapses at start of simulation
    weight=5,
    # Use this delay when creating a new synapse
    initial_delay=5,
    # Use this weight for synapses at the start of simulation
    delay=5,
    # Maximum allowed fan-in per target-layer neuron
    s_max=32,
    # Frequency of rewiring in Hz
    f_rew=10 ** 4
)

plastic_projection = sim.Projection(
    pre_pop, post_pop,
    sim.FixedProbabilityConnector(0.),  # No initial connections
    synapse_type=structure_model_without_stdp,
    label="structurally_plastic_projection"
)

sim.run(simtime)

pre_neo = pre_pop.get_data(variables=["spikes"])
pre_spikes = pre_neo.segments[0].spiketrains

post_neo = post_pop.get_data(variables=["spikes"])
post_spikes = post_neo.segments[0].spiketrains

print(plastic_projection.get('weight', format="list"))

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
