import matplotlib.pyplot as plt
import pyNN.utility.plotting as plotting
import spynnaker8 as sim

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
                           sim.SpikeSourcePoisson(rate=e_rate, start=0,
                                                  duration=simtime),
                           label="expoisson")
INoisePost = sim.Population(pop_size,
                            sim.SpikeSourcePoisson(rate=e_rate, start=0,
                                                   duration=simtime),
                            label="expoisson")

# +-------------------------------------------------------------------+
# | Creation of connections                                           |
# +-------------------------------------------------------------------+

# Connection parameters
JEE = 3.

# Connection type between noise poisson generator and excitatory populations
ee_connector = sim.OneToOneConnector()
ee_synapse_type = sim.StaticSynapse(weight=JEE * 0.05)

# Noise projections
sim.Projection(INoisePre, pre_pop, ee_connector, ee_synapse_type,
               receptor_type='excitatory')
sim.Projection(INoisePost, post_pop, ee_connector, ee_synapse_type,
               receptor_type='excitatory')

# Additional Inputs projections
for i in range(len(IAddPre)):
    sim.Projection(IAddPre[i], pre_pop, ee_connector, ee_synapse_type,
                   receptor_type='excitatory')
for i in range(len(IAddPost)):
    sim.Projection(IAddPost[i], post_pop, ee_connector, ee_synapse_type,
                   receptor_type='excitatory')

# Plastic Connections between pre_pop and post_pop
stdp_model = sim.STDPMechanism(
    timing_dependence=sim.SpikePairRule(tau_plus=20., tau_minus=20.0,
                                        A_plus=0.02, A_minus=0.02),
    weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.9)
)

plastic_projection = sim.Projection(
    pre_pop, post_pop, sim.FixedProbabilityConnector(p_connect=0.5),
    synapse_type=sim.STDPMechanism(
        timing_dependence=sim.SpikePairRule(tau_plus=20., tau_minus=20.0,
                                            A_plus=0.02, A_minus=0.02),
        weight_dependence=sim.AdditiveWeightDependence(w_min=0, w_max=0.9)
    )
)

# +-------------------------------------------------------------------+
# | Simulation and results                                            |
# +-------------------------------------------------------------------+

# Record neurons' potentials
pre_pop.record("v")
post_pop.record("v")

# Record spikes
pre_pop.record("spikes")
post_pop.record("spikes")

# Run simulation
sim.run(simtime)

print("Weights:", plastic_projection.get(attribute_names='weight',
                                         format='list', gather=True,
                                         with_address=True))

pre_spikes = pre_pop.get_data('spikes')
post_spikes = post_pop.get_data('spikes')

plotting.Figure(
    plotting.Panel(
        pre_spikes.segments[0].spiketrains,
        post_spikes.segments[0].spiketrains,
        yticks=True, markersize=5, xlim=(0, simtime),
        line_properties=[{"color": "r"}, {"color": "b"}]
        ),
    title="Learning example",
    annotations="Simulated with {}".format(sim.name())
)

plt.show()

# End simulation on SpiNNaker
sim.end()
