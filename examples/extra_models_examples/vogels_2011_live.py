import spynnaker8 as sim


# -------------------------------------------------------------------
# This example uses the sPyNNaker implementation of the inhibitory
# Plasticity rule developed by Vogels, Sprekeler, Zenke et al (2011)
# To reproduce the experiment from their paper
# -------------------------------------------------------------------
# Population parameters
model = sim.IF_curr_exp
cell_params = {
    'cm': 0.2,          # nF
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

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)

# Reduce number of neurons to simulate on each core
sim.set_number_of_neurons_per_core(sim.IF_curr_exp, 100)

# Create excitatory and inhibitory populations of neurons
ex_pop = sim.Population(NUM_EXCITATORY, model(**cell_params),
                        label="Excitatory")
in_pop = sim.Population(NUM_EXCITATORY / 4, model(**cell_params),
                        label="Inhibitory")

# Record excitatory spikes
ex_pop.record('spikes')

# Make excitatory->inhibitory projections
sim.Projection(ex_pop, in_pop, sim.FixedProbabilityConnector(0.02),
               receptor_type='excitatory',
               synapse_type=sim.StaticSynapse(weight=0.03))
sim.Projection(ex_pop, ex_pop, sim.FixedProbabilityConnector(0.02),
               receptor_type='excitatory',
               synapse_type=sim.StaticSynapse(weight=0.03))

# Make inhibitory->inhibitory projections
sim.Projection(in_pop, in_pop, sim.FixedProbabilityConnector(0.02),
               receptor_type='inhibitory',
               synapse_type=sim.StaticSynapse(weight=-0.3))

# Build inhibitory plasticity  model
stdp_model = sim.STDPMechanism(
    timing_dependence=sim.extra_models.Vogels2011Rule(alpha=0.12, tau=20.0,
                                                      A_plus=0.0005),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0.0, w_max=1.0))

# Make inhibitory->excitatory projections
sim.Projection(in_pop, ex_pop, sim.FixedProbabilityConnector(0.02),
               receptor_type='inhibitory',
               synapse_type=stdp_model)

# Activate live output for excitatory spikes
sim.external_devices.activate_live_output_for(ex_pop)
sim.external_devices.activate_live_output_for(in_pop)

# Run simulation
sim.run(5000)

# End simulation on SpiNNaker
sim.end()
