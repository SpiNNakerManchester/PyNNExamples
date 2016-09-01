"""
Simple test for STDP :

   Reproduces a classical plasticity experiment of plasticity induction by
pre/post synaptic pairing specifically :

 * At the begining of the simulation, "n_stim_test" external stimulations of
   the "pre_pop" (presynaptic) population do not trigger activity in the
   "post_pop" (postsynaptic) population.

 * Then the presynaptic and postsynaptic populations are stimulated together
   "n_stim_pairing" times by an external source so that the "post_pop"
   population spikes 10ms after the "pre_pop" population.

 * Ater that period, only the "pre_pop" population is externally stimulated
   "n_stim_test" times, but now it should trigger activity in the "post_pop"
   population (due to STDP learning)

Run as :

   $ ./stdp_example

This example requires that the NeuroTools package is installed
(http://neuralensemble.org/trac/NeuroTools)

Authors : Catherine Wacongne < catherine.waco@gmail.com >
          Xavier Lagorce < Xavier.Lagorce@crans.org >

April 2013
"""
import pylab
try:
    import pyNN.spiNNaker as sim
except Exception as e:
    import spynnaker.pyNN as sim

# SpiNNaker setup
sim.setup(timestep=1.0, min_delay=1.0, max_delay=10.0)

# +-------------------------------------------------------------------+
# | General Parameters                                                |
# +-------------------------------------------------------------------+

# Population parameters
model = sim.IF_curr_exp

cell_params = {'cm': 0.25,
               'i_offset': 0.0,
               'tau_m': 20.0,
               'tau_refrac': 2.0,
               'tau_syn_E': 5.0,
               'tau_syn_I': 5.0,
               'v_reset': -70.0,
               'v_rest': -65.0,
               'v_thresh': -50.0
               }


# Other simulation parameters
e_rate = 80
in_rate = 300

n_stim_test = 5
n_stim_pairing = 20
dur_stim = 20

pop_size = 40

ISI = 90.
start_test_pre_pairing = 200.
start_pairing = 1500.
start_test_post_pairing = 700.

simtime = (start_pairing + start_test_post_pairing
           + ISI * (n_stim_pairing + n_stim_test) + 550.)

# Initialisations of the different types of populations
IAddPre = []
IAddPost = []

# +-------------------------------------------------------------------+
# | Creation of neuron populations                                    |
# +-------------------------------------------------------------------+

# Neuron populations
pre_pop = sim.Population(pop_size, model, cell_params)
post_pop = sim.Population(pop_size, model, cell_params)

# Test of the effect of activity of the pre_pop population on the post_pop
# population prior to the "pairing" protocol : only pre_pop is stimulated
for i in range(n_stim_test):
    IAddPre.append(sim.Population(pop_size,
                                  sim.SpikeSourcePoisson,
                                  {'rate': in_rate,
                                   'start': start_test_pre_pairing + ISI * (i),
                                   'duration': dur_stim
                                   }))

# Pairing protocol : pre_pop and post_pop are stimulated with a 10 ms
# difference
for i in range(n_stim_pairing):
    IAddPre.append(sim.Population(pop_size,
                                  sim.SpikeSourcePoisson,
                                  {'rate': in_rate,
                                   'start': start_pairing + ISI * (i),
                                   'duration': dur_stim
                                   }))
    IAddPost.append(sim.Population(pop_size,
                                   sim.SpikeSourcePoisson,
                                   {'rate': in_rate,
                                    'start': start_pairing + ISI * (i) + 10.,
                                    'duration': dur_stim
                                    }))

# Test post pairing : only pre_pop is stimulated (and should trigger activity
# in Post)
for i in range(n_stim_test):
    IAddPre.append(sim.Population(pop_size,
                                  sim.SpikeSourcePoisson,
                                  {'rate': in_rate,
                                   'start': (start_pairing
                                             + ISI * (n_stim_pairing)
                                             + start_test_post_pairing
                                             + ISI * (i)),
                                   'duration': dur_stim
                                   }))

# Noise inputs
INoisePre = sim.Population(pop_size,
                           sim.SpikeSourcePoisson,
                           {'rate': e_rate, 'start': 0, 'duration': simtime},
                           label="expoisson")
INoisePost = sim.Population(pop_size,
                            sim.SpikeSourcePoisson,
                            {'rate': e_rate, 'start': 0, 'duration': simtime},
                            label="expoisson")

# +-------------------------------------------------------------------+
# | Creation of connections                                           |
# +-------------------------------------------------------------------+

# Connection parameters
JEE = 3.

# Connection type between noise poisson generator and excitatory populations
ee_connector = sim.OneToOneConnector(weights=JEE * 0.05)

# Noise projections
sim.Projection(INoisePre, pre_pop, ee_connector, target='excitatory')
sim.Projection(INoisePost, post_pop, ee_connector, target='excitatory')

# Additional Inputs projections
for i in range(len(IAddPre)):
    sim.Projection(IAddPre[i], pre_pop, ee_connector, target='excitatory')
for i in range(len(IAddPost)):
    sim.Projection(IAddPost[i], post_pop, ee_connector, target='excitatory')

# Plastic Connections between pre_pop and post_pop
stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=20., tau_minus=20.0,
                                        nearest=True),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.9,
                                                   A_plus=0.02, A_minus=0.02)
)

plastic_projection = sim.Projection(
    pre_pop, post_pop, sim.FixedProbabilityConnector(p_connect=0.5),
    synapse_dynamics=sim.SynapseDynamics(slow=stdp_model)
)

# +-------------------------------------------------------------------+
# | Simulation and results                                            |
# +-------------------------------------------------------------------+

# Record neurons' potentials
pre_pop.record_v()
post_pop.record_v()

# Record spikes
pre_pop.record()
post_pop.record()

# Run simulation
sim.run(simtime)

print("Weights:", plastic_projection.getWeights())

pre_spikes = pre_pop.getSpikes(compatible_output=True)
post_spikes = post_pop.getSpikes(compatible_output=True)

pylab.figure()
pylab.xlim((0, simtime))
pylab.plot([i[1] for i in pre_spikes], [i[0] for i in pre_spikes], "r.")
pylab.plot([i[1] for i in post_spikes], [i[0] for i in post_spikes], "b.")
pylab.xlabel('Time/ms')
pylab.ylabel('spikes')

pylab.show()

# End simulation on SpiNNaker
sim.end()
