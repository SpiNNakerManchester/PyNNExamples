"""
Simple test for neuromodulated STDP.
Two pre-synaptic spikes are added, at times 1500 and 2400ms.
Post-synaptic neuron is stimulated at 1502 and fires at time 1503ms.
Dendritic delay is 1ms so post-synaptic time is at 1504ms when processed in
STDP. Dopamine neuron spikes at 1600+1ms (Also added dendritic delay).
Calculating weight change in this scenario, according to equations in the
Izhikevich 2007 paper*, gives us the weight change of 10.0552710...
*https://www.ncbi.nlm.nih.gov/pubmed/17220510
Simulation from SpiNNaker gives us the weight change of 10.0087890625.
Some inaccuracy occurs due to precision loss in s5.11 fixed point format
used in STDP traces and exp LUTs. Also, due to long timing constants, exp
LUTs are discretized further by TAU_C_SHIFT and TAU_D_SHIFT to be able to
fit them into memory, adding another level of inaccuracy. Finally, some more
accuracy may be lost due to weight scaling.
"""

try:
    import pyNN.spiNNaker as sim
except Exception:
    import spynnaker8 as sim


from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

timestep = 1.0
duration = 10000


rewards = [x for x in range(0, duration, 250)]
# out_stim = [x for x in range(1000, 1010)] + \
# out_stim = [x for x in range(1000, 1010)] + \
#            [x for x in range(2000, 2020)] + \
#            [x for x in range(3000, 3100)]

# Main parameters from Izhikevich 2007 STDP paper
t_pre = [1500, 2400]    # Pre-synaptic neuron times
t_post = [1502]         # Post-synaptic neuron stimuli time
t_dopamine = [1020, 2030, 3150]     # Dopaminergic neuron spike times
tau_c = 1000            # Eligibility trace decay time constant.
tau_d = 200             # Dopamine trace decay time constant.
DA_concentration = 0.1  # Dopamine trace step increase size

# Initial weight
rewarded_syn_weight = 0.0

cell_params = {
    'cm': 0.3,
    'i_offset': 0.0,
    'tau_m': 10.0,
    'tau_refrac': 4.0,
    'tau_syn_E': 1.0,
    'tau_syn_I': 1.0,
    'v_reset': -70.0,
    'v_rest': -65.0,
    'v_thresh': -55.4}

sim.setup(timestep=timestep)

pre_pop = sim.Population(2, sim.SpikeSourcePoisson, {                                  # Input population of two neurons, 1 per target sound
    'rate': [10, 0]})

# Create a population of dopaminergic neurons for reward
reward_pop = sim.Population(1, sim.SpikeSourceArray, {                               # Comes from stroking sensor
    'spike_times': rewards},
    label='reward'
    )

# Stimulus for post synaptic population
# post_stim = sim.Population(2, sim.SpikeSourceArray, {                                # Stimulation for output neurons (to enable post spiking for STDP-based learning)
#     'spike_times': [out_stim, []]
#     }
#     )
post_stim = sim.Population(2, sim.SpikeSourcePoisson, {                                # Stimulation for output neurons (to enable post spiking for STDP-based learning)
#     'rate': [10, 0]
    'rate': [1, 1]
#     'rate': [0, 10]
    }
    )

# Create post synaptic population which will be modulated by DA concentration.
post_pop = sim.Population(2, sim.IF_curr_exp_izhikevich_neuromodulation,             # Output neurons (1 for each action)
                          cell_params, label='post_pop')

# Create STDP dynamics with neuromodulation
synapse_dynamics = sim.STDPMechanism(
    timing_dependence=sim.IzhikevichNeuromodulation(
        tau_plus=10, tau_minus=12,
        A_plus=1, A_minus=1,
        tau_c=1000, tau_d=200),
    weight_dependence=sim.MultiplicativeWeightDependence(
        w_min=0, w_max=20),
    weight=0.0,
    neuromodulation=True)

# Create dopaminergic connection
reward_projection = sim.Projection(
    post_stim, post_pop,
#     sim.AllToAllConnector(),
    sim.FromListConnector([(0, 1, DA_concentration, 1)]),
    synapse_type=sim.StaticSynapse(weight=DA_concentration),
    receptor_type='reward', label='reward synapses')

# Stimulate post-synaptic neuron
sim.Projection(
    post_stim, post_pop,
    sim.OneToOneConnector(),
    synapse_type=sim.StaticSynapse(weight=6),
    receptor_type='excitatory')

# Create a plastic connection between pre and post neurons
plastic_projection = sim.Projection(
    pre_pop, post_pop,
    sim.AllToAllConnector(),
    synapse_type=synapse_dynamics,
    receptor_type='excitatory', label='Pre-post projection')

lateral_inhib = sim.Projection(
    pre_pop, post_pop,
    sim.AllToAllConnector(allow_self_connections=False),
    synapse_type=sim.StaticSynapse(weight=3),
    receptor_type='inhibitory', label='Pre-post projection')


pre_pop.record('all')
reward_pop.record('all')
post_pop.record('all')
post_stim.record('all')

sim.run(duration)

pre_data = pre_pop.get_data()
reward_data = reward_pop.get_data()
post_data = post_pop.get_data()
post_stim_data = post_stim.get_data()

F = Figure(
    # plot data for postsynaptic neuron
#     Panel(post_pop.segments[0].filter(name='v')[0],
#           ylabel="Membrane potential (mV)",
#           data_labels=[post_pop.label], yticks=True, xlim=(0, runtime)
#           ),
#     Panel(post_pop.segments[0].filter(name='gsyn_exc')[0],
#           ylabel="gsyn excitatory (mV)",
#           data_labels=[post_pop.label], yticks=True, xlim=(0, runtime)
#           ),
#     Panel(post_pop.segments[0].filter(name='gsyn_inh')[0],
#           ylabel="gsyn inhibitory (mV)",
#           data_labels=[pop_exc.label], yticks=True, xlim=(0, runtime)
#           ),
    Panel(pre_data.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, duration)
          ),
    Panel(reward_data.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, duration)
          ),
    Panel(post_stim_data.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, duration)
          ),
    Panel(post_data.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, duration)
          ),
    )

# End simulation on SpiNNaker
print("Final weight: " + repr(plastic_projection.get('weight', 'list')))

sim.end()

plt.show()
