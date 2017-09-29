"""
Simple Associative Memory
"""
import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
from pyNN.random import NumpyRNG, RandomDistribution

p.setup(timestep=1.0, min_delay=1.0, max_delay=15.0)
p.set_number_of_neurons_per_core(p.IF_curr_exp, 100)

nSourceNeurons = 1  # number of input (excitatory) neurons
nExcitNeurons = 1   # number of excitatory neurons in the recurrent memory
nInhibNeurons = 10  # number of inhibitory neurons in the recurrent memory
nTeachNeurons = 1
nNoiseNeurons = 10
runTime = 3200

input_cell_params = {
    'cm': 0.25,        # nF
    'i_offset': 5.0,
    'tau_m': 10.0,
    'tau_refrac': 2.0,
    'tau_syn_E': 0.5,
    'tau_syn_I': 0.5,
    'v_reset': -65.0,
    'v_rest': -65.0,
    'v_thresh': -64.4}

cell_params_lif = {
    'cm': 0.25,        # nF was 0.25
    'i_offset': 0.0,
    'tau_m': 10.0,
    'tau_refrac': 2.0,
    'tau_syn_E': 0.5,
    'tau_syn_I': 0.5,
    'v_reset': -70.0,
    'v_rest': -70.0,
    'v_thresh': -50.0}

populations = list()
projections = list()

stimulus = 0
inhib = 1
excit = 2
teacher = 3
noise = 3
decoy = 3

rate = 100
stim_dur = 150
noise_weight = 0.04 * 4.0/3.0
weight_to_force_firing = 15.0
baseline_excit_weight = 2.0
baseline_inhib_weight = 1.0
# weight_to_spike = 4.7
# weight_recurrent = 1.0
# weight_to_inhib = 0.5
# inhib_weight    = 0.0
# excit_in_delay  = 3.0
# inhib_in_delay  = 1.0
# inhib_out_delay = 1.0
# p_connect = 1.0
# conn_prob = 0.75
p_exc2exc = 0.10
p_exc2inh = 0.10
p_inh2exc = 0.17
p_to_inhib_connect = 1.0
p_from_inhib_connect = 1.0

spikes0 = list()
teachingSpikes = list()
for i in range(runTime/40):
    spikes0.append(i*40)
for i in range(runTime/80):
    teachingSpikes.append(i*40+5+120)

# spikes0 = [0, 40, 80, 120, 160, 200, 240, 280, 320, 360, 400, 440, 480, 520,
#            560, 600, 640, 680]
spikes1 = [10, 50, 90, 130, 170, 210, 250, 290]
spikes2 = [20, 60, 100, 140, 180, 220, 260, 300]
spikes3 = [30, 70, 110, 150, 190, 230, 270, 310]
spikes4 = []
arrayEntries = []
for i in range(nSourceNeurons):
    newEntry = []
    for j in range(len(spikes0)):
        newEntry.append(spikes0[j] + i*40.0/100.0)
    arrayEntries.append(newEntry)
spikeArray = {'spike_times': arrayEntries}

teachlist = list()
for i in range(nSourceNeurons):
    teachlist.append(teachingSpikes)
teachingSpikeArray = {'spike_times': teachlist}
populations.append(p.Population(nSourceNeurons,
                                p.SpikeSourceArray(**spikeArray),
                                label='excit_pop_ss_array'))       # 0
populations.append(p.Population(nInhibNeurons,
                                p.IF_curr_exp(**cell_params_lif),
                                label='inhib_pop'))                # 1
populations.append(p.Population(nExcitNeurons,
                                p.IF_curr_exp(**cell_params_lif),
                                label='excit_pop'))                # 2
populations.append(p.Population(nTeachNeurons,
                                p.SpikeSourceArray(**teachingSpikeArray),
                                label='teaching_ss_array'))        # 3

stdp_model = p.STDPMechanism(
    timing_dependence=p.extra_models.RecurrentRule(
        accumulator_depression=-6, accumulator_potentiation=3,
        mean_pre_window=10.0, mean_post_window=10.0, dual_fsm=True,
        A_plus=0.2, A_minus=0.2),
    weight_dependence=p.MultiplicativeWeightDependence(w_min=0.0, w_max=16.0),
    weight=baseline_excit_weight, delay=1)

rng = NumpyRNG(seed=1)
ext_delay_distr = RandomDistribution('normal', (1.5, 0.75), rng=rng)
inh_delay_distr = RandomDistribution('normal', (0.75, 0.375), rng=rng)
delay_distr_recurrent = RandomDistribution('uniform', [2.0, 8.0], rng=rng)
weight_distr_ffwd = RandomDistribution('uniform', [0.5, 1.25], rng=rng)
weight_distr_recurrent = RandomDistribution('uniform', [0.1, 0.2], rng=rng)

projections.append(
    p.Projection(populations[stimulus], populations[excit],
                 p.AllToAllConnector(), synapse_type=stdp_model))

projections.append(
    p.Projection(populations[teacher], populations[excit],
                 p.OneToOneConnector(), receptor_type='excitatory',
                 synapse_type=p.StaticSynapse(
                     weight=weight_to_force_firing, delay=1)))
# projections.append(
#     p.Projection(populations[stimulus], populations[inhib],
#                  p.FixedProbabilityConnector(p_connect=p_exc2inh,
#                                              weights=baseline_excit_weight,
#                                              delays=inh_delay_distr),
#                  target='excitatory'))

# populations[stimulus].record(['spikes'])

populations[inhib].record(['v', 'spikes'])
populations[excit].record(['v', 'spikes'])

# populations[teacher].record(['spikes'])

# populations[noise].record(['v', 'spikes')

p.run(runTime)

final_weights = projections[0].get('weight', 'list', with_address=False)
print "Final weights: ", final_weights

# myDelays = projections[0].getDelays()
# total=0.0
# count = len(myDelays)
# for i in range(count):
#    total = total + myDelays[i]
# print "Average: ", total*1.0/count
# print "Delays:\n", myDelays

# myDelays = projections[1].getDelays()
# print "Delays:\n", myDelays
# myDelaysRecurrent = projections[2].getDelays(format='list')
# print "Recurrent delays:\n", myDelaysRecurrent
# myWeights= projections[0].getWeights()
# print "Weights:\n", myWeights
# myWeightsRecurr= projections[2].getWeights()
# print "Recurrent weights:\n", myWeightsRecurr

v = None
gsyn = None
spikes = None

v = populations[excit].get_data('v')
# spikesStim = populations[stimulus].get_data('spikes')
spikes = populations[excit].get_data('spikes')
vInhib = populations[inhib].get_data('v')
spikesInhib = populations[inhib].get_data('spikes')
# spikesTeach = populations[teacher].get_data('spikes')
# spikesNoise = populations[noise].get_data('spikes')

Figure(
    # plot of the neuron spike times
    Panel(spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runTime)),
    # membrane potential of the neurons
    Panel(v.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[populations[excit].label], yticks=True,
          xlim=(0, runTime), xticks=True),
    title="Simple associative memory: spikes and membrane potential",
    annotations="Simulated with {}".format(p.name())
)
plt.show()

p.end()
