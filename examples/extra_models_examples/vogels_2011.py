import spynnaker8 as sim
import numpy
import pylab
from spynnaker8.utilities import neo_convertor

# -------------------------------------------------------------------
# This example uses the sPyNNaker implementation of the inhibitory
# Plasticity rule developed by Vogels, Sprekeler, Zenke et al (2011)
# To reproduce the experiment from their paper
# -------------------------------------------------------------------
# Population parameters
model = sim.IF_curr_exp
cell_params = {
    'cm': 0.2,         # nF
    'i_offset': 0.2,
    'tau_m': 20.0,
    'tau_refrac': 5.0,
    'tau_syn_E': 5.0,
    'tau_syn_I': 10.0,
    'v_reset': -60.0,
    'v_rest': -60.0,
    'v_thresh': -50.0}


# How large should the population of excitatory neurons be?
# (Number of inhibitory neurons is proportional to this)
NUM_EXCITATORY = 2000


# Function to build the basic network - pass in the stdp_model
def build_network(stdp_model):
    # SpiNNaker setup
    sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)

    # Reduce number of neurons to simulate on each core
    sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 10)

    # Create excitatory and inhibitory populations of neurons
    ex_pop = sim.Population(NUM_EXCITATORY, model(**cell_params))
    in_pop = sim.Population(NUM_EXCITATORY / 4, model(**cell_params))

    # Record excitatory spikes
    ex_pop.record(['spikes'])

    # Make excitatory->inhibitory projections
    sim.Projection(ex_pop, in_pop,
                   sim.FixedProbabilityConnector(0.02),
                   receptor_type='excitatory',
                   synapse_type=sim.StaticSynapse(weight=0.03))
    sim.Projection(ex_pop, ex_pop,
                   sim.FixedProbabilityConnector(0.02),
                   receptor_type='excitatory',
                   synapse_type=sim.StaticSynapse(weight=0.03))

    # Make inhibitory->inhibitory projections
    sim.Projection(in_pop, in_pop,
                   sim.FixedProbabilityConnector(0.02),
                   receptor_type='inhibitory',
                   synapse_type=sim.StaticSynapse(weight=-0.3))

    # Make inhibitory->excitatory projections
    ie_projection = sim.Projection(
        in_pop, ex_pop, sim.FixedProbabilityConnector(0.02),
        receptor_type='inhibitory', synapse_type=stdp_model)

    return ex_pop, ie_projection


# Build static network
static_ex_pop, _ = build_network(None)

# Run for 1s
sim.run(1000)

# Get static spikes
static_spikes = static_ex_pop.get_data('spikes')
static_spikes_numpy = neo_convertor.convert_spikes(static_spikes)

# Build inhibitory plasticity  model
stdp_model = sim.STDPMechanism(
    timing_dependence=sim.extra_models.Vogels2011Rule(alpha=0.12, tau=20.0,
                                                      A_plus=0.05),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=1.0))

# Build plastic network
plastic_ex_pop, plastic_ie_projection = build_network(stdp_model)

# Run simulation
sim.run(10000)

# Get plastic spikes and save to disk
plastic_spikes = plastic_ex_pop.get_data('spikes')
plastic_spikes_numpy = neo_convertor.convert_spikes(plastic_spikes)
numpy.save("plastic_spikes.npy", plastic_spikes_numpy)

plastic_weights = plastic_ie_projection.get('weight', 'array')
# Weights(format="array")
mean_weight = numpy.average(plastic_weights)
print "Mean learnt ie weight:%f" % mean_weight

# Create plot
fig, axes = pylab.subplots(3)

# Plot last 200ms of static spikes (to match Brian script)
axes[0].set_title("Excitatory raster without inhibitory plasticity")
axes[0].scatter(static_spikes_numpy[:, 1],
                static_spikes_numpy[:, 0], s=2, color="blue")
axes[0].set_xlim(800, 1000)
axes[0].set_ylim(0, NUM_EXCITATORY)

# Plot last 200ms of plastic spikes (to match Brian script)
axes[1].set_title("Excitatory raster with inhibitory plasticity")
axes[1].scatter(plastic_spikes_numpy[:, 1],
                plastic_spikes_numpy[:, 0], s=2, color="blue")
axes[1].set_xlim(9800, 10000)
axes[1].set_ylim(0, NUM_EXCITATORY)

# Plot rates
binsize = 10
bins = numpy.arange(0, 10000 + 1, binsize)
plastic_hist, _ = numpy.histogram(plastic_spikes_numpy[:, 1], bins=bins)
plastic_rate = plastic_hist * (1000.0/binsize) * (1.0/NUM_EXCITATORY)
axes[2].set_title("Excitatory rates with inhibitory plasticity")
axes[2].plot(bins[0:-1], plastic_rate, color="red")
axes[2].set_xlim(9800, 10000)
axes[2].set_ylim(0, 20)

# Show figures
pylab.show()

# End simulation on SpiNNaker
sim.end()
