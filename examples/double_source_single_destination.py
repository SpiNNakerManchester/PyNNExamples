import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

runtime = 100
value = 4
nNeurons = 64*value
p.setup(timestep=0.1)
#p.set_number_of_neurons_per_core(p.IF_curr_exp_plastic, nNeurons / 2)

dest_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0,
                   }

input_params_lif = {'cm': 0.25,
                   'i_offset': 0.0,
                   'tau_m': 20.0,
                   'tau_refrac': 2.0,
                   'tau_syn_E': 5.0,
                   'tau_syn_I': 5.0,
                   'v_reset': -70.0,
                   'v_rest': -65.0,
                   'v_thresh': -50.0,
                   'v' : -40.0,
                   }

# weight_to_spike = 0.035
weight_to_spike = 0.01
delay = 3

destination = p.Population(64, p.IF_curr_exp(**dest_params_lif), label='population')
input_exc = p.Population(nNeurons, p.IF_curr_exp(**input_params_lif), label='input1')
input_inh = p.Population(nNeurons, p.IF_curr_exp(**input_params_lif), label='input2')

p.Projection(input_exc, destination, p.FixedProbabilityConnector(p_connect=0.1),
             p.StaticSynapse(weight=weight_to_spike,delay=2), receptor_type="excitatory")
p.Projection(input_inh, destination, p.FixedProbabilityConnector(p_connect=0.1),
             p.StaticSynapse(weight=weight_to_spike, delay=4), receptor_type="inhibitory")

destination.record(['v', 'spikes', 'gsyn_exc', 'gsyn_inh'])
input_exc.record(['v', 'spikes'])
input_inh.record(['v', 'spikes'])

p.run(runtime)

v = destination.get_data('v')
spikes = destination.get_data('spikes')
ge = destination.get_data('gsyn_exc')
gi = destination.get_data('gsyn_inh')

ve = input_exc.get_data('v')
spikese = input_exc.get_data('spikes')

vi = input_inh.get_data('v')
spikesi = input_inh.get_data('spikes')

figure_filename = "results.png"
Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime), xticks=True),
    # membrane potential of the postsynaptic neuron
    Panel(v.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[destination.label], yticks=True, xlim=(0, runtime), xticks=True),
    Panel(ge.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[destination.label], yticks=True, xlim=(0, runtime), xticks=True),
    Panel(gi.segments[0].filter(name='gsyn_inh')[0],
          ylabel="gsyn inhibitory (mV)",
          data_labels=[destination.label], yticks=True, xlim=(0, runtime), xticks=True),
    title="Single neuron",
    annotations="Simulated with {}".format(p.name())
)

plt.show()

Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(spikese.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime), xticks=True),
    # membrane potential of the postsynaptic neuron
    Panel(ve.segments[0].filter(name='v')[0],
          ylabel="Membrane potential excitatory (mV)",
          data_labels=[destination.label], yticks=True, xlim=(0, runtime), xticks=True),
    Panel(spikesi.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, runtime), xticks=True),
    # membrane potential of the postsynaptic neuron
    Panel(vi.segments[0].filter(name='v')[0],
          ylabel="Membrane potential inhibitory (mV)",
          data_labels=[destination.label], yticks=True, xlim=(0, runtime), xticks=True),
    title="Single neuron",
    annotations="Simulated with {}".format(p.name())
)


# for n in range(len(spikes.segments[0].spiketrains)):
#     print "Neuron (exc): " + str(n) + " spiked at timestep: " + str(spikese.segments[0].spiketrains[n])
#     print "Neuron (inh): " + str(n) + " spiked at timestep: " + str(spikesi.segments[0].spiketrains[n])
#     print "\n"

plt.show()
print(figure_filename)

p.end()