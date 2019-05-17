"""
Simple test for neuromodulated STDP.
Two pre-synaptic spikes are added, at times 1500 and 2400ms.
Post-synaptic neuron is stimulated to fire at time 1503ms. Dendritic delay is
1ms so post-synaptic time is at 1504ms when processed in STDP.
Dopamine neuron spikes at 1600+1ms (Also added dendritic delay).
Calculating weight change in this scenario, according to equations in the
Izhikevich 2007 paper*, gives us the weight change of 10.305624...
*https://www.ncbi.nlm.nih.gov/pubmed/17220510
Simulation from SpiNNaker gives us the weight change of 10.0087890625.
Some inaccuracy occurs due to precision loss in s5.11 fixed point format
used in STDP calculations and exp LUTs. Thus the error is smaller for smaller
timing constants.
"""

try:
    import pyNN.spiNNaker as sim
except Exception:
    import spynnaker8 as sim

timestep = 1.0
duration = 3000

# Main parameters from Izhikevich 2007 STDP paper
t_pre = [1500, 2400]   # Pre-synaptic neuron times
t_post = [1502]        # Post-synaptic neuron stimuli time
t_dopamine = [1600]    # Dopaminergic neuron spike times
tau_c = 1000           # Eligibility trace decay time constant.
tau_d = 200            # Dopamine trace decay time constant.
DA_concentration = 0.1 # Dopamine trace step increase size

# Initial weight
rewarded_syn_weight = 0.0

cell_params = {'cm': 0.3,
			   'i_offset': 0.0,
			   'tau_m': 10.0,
			   'tau_refrac': 4.0,
			   'tau_syn_E': 1.0,
			   'tau_syn_I': 1.0,
			   'v_reset': -70.0,
			   'v_rest': -65.0,
			   'v_thresh': -55.4
			  }

sim.setup(timestep=timestep)

pre_pop = sim.Population(1, sim.SpikeSourceArray,
	{'spike_times': t_pre})

# Create a population of dopaminergic neurons for reward
reward_pop = sim.Population(1, sim.SpikeSourceArray,
	{'spike_times': t_dopamine}, label='reward')

# Stimulus for post synaptic population
post_stim = sim.Population(1, sim.SpikeSourceArray,
	{'spike_times': t_post})

# Create post synaptic population which will be modulated by DA concentration.
post_pop = sim.Population(1, sim.IF_curr_exp_izhikevich_neuromodulation,
	 cell_params, label='post1')

# Create STDP dynamics with neuromodulation
synapse_dynamics = sim.STDPMechanism(
    timing_dependence=sim.IzhikevichNeuromodulation(
                        tau_plus=10, tau_minus=12,
                        A_plus=1, A_minus=1,
						tau_c=1000, tau_d=200),
						weight_dependence=sim.MultiplicativeWeightDependence(
						w_min=0, w_max=20),
                        weight=0.0,
						neuromodulation=True);

# Create dopaminergic connection
reward_projection = sim.Projection(reward_pop, post_pop,
	sim.AllToAllConnector(),
    synapse_type=sim.StaticSynapse(weight=DA_concentration),
	receptor_type='reward', label='reward synapses')

# Stimulate post-synaptic neuron
sim.Projection(post_stim, post_pop,
	sim.AllToAllConnector(),
    synapse_type=sim.StaticSynapse(weight=6),
	receptor_type='excitatory')

# Create a plastic connection between pre and post neurons
plastic_projection = sim.Projection(pre_pop, post_pop,
	sim.AllToAllConnector(),
	synapse_type=synapse_dynamics,
	receptor_type='excitatory', label='Pre-post projection')

sim.run(duration)

# End simulation on SpiNNaker
print "Final weight: " + repr(plastic_projection.get('weight', 'list'))

sim.end()
