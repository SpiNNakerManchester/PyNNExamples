"""
Spiking Elementary Motion Detector (sEMD) example
See https://www.cit-ec.de/en/nbs/spiking-insect-vision for more details
"""

# imports
import spynnaker8 as p
import datetime
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

# parameters
datum = datetime.datetime.now()

step = 0.1
p.setup(timestep=step)
n_neurons = 1
run_time = 150
cm = 0.25
i_offset = 0.0
tau_m = 20.0
tau_refrac = 1.0
current_decay = tau_syn_E = tau_syn_I = 30
v_reset = -85.0
v_rest = -65.0
v_thresh = -50.0

weight = 1
delay1 = 0.1
delay2 = 2.0

cell_params_lif = {'cm': cm, 'i_offset': i_offset, 'tau_m': tau_m,
                   'tau_refrac': tau_refrac, 'tau_syn_E': current_decay,
                   'tau_syn_I': current_decay, 'v_reset': v_reset,
                   'v_rest': v_rest, 'v_thresh': v_thresh}


# neuron populations
sEMD = p.Population(1, p.IF_curr_exp_sEMD(**cell_params_lif), label="sEMD")
spikeArray = {'spike_times': [[0]]}
input_first = p.Population(1, p.SpikeSourceArray(**spikeArray),
                           label="input_first")
input_second = p.Population(1, p.SpikeSourceArray(**spikeArray),
                            label="input_second")

# projections
p.Projection(input_first, sEMD,
             p.OneToOneConnector(),
             receptor_type="excitatory",
             synapse_type=p.StaticSynapse(weight=weight, delay=delay1))
p.Projection(input_second, sEMD,
             p.OneToOneConnector(),
             receptor_type="inhibitory",
             synapse_type=p.StaticSynapse(weight=weight, delay=delay2))

# records
sEMD.record(['v', 'gsyn_exc', 'gsyn_inh', 'spikes'])

# run
p.run(run_time)

# get data
spikes = sEMD.get_data(['spikes'])  # read spikes
v = sEMD.get_data(['v'])  # read membrane voltage
current_exc = sEMD.get_data(['gsyn_exc'])  # read excitatory
current_inh = sEMD.get_data(['gsyn_inh'])  # read inhibitory

print datum

# plots
Figure(
    # raster plot of the presynaptic neuron spike times
    Panel(spikes.segments[0].spiketrains,
          yticks=True, markersize=0.2, xlim=(0, run_time)),
    # membrane potential of the postsynaptic neuron
    Panel(v.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[sEMD.label], yticks=True, xlim=(0, run_time)),
    Panel(current_exc.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[sEMD.label], yticks=True, xlim=(0, run_time)),
    Panel(current_inh.segments[0].filter(name='gsyn_inh')[0],
          xlabel="Time (ms)", xticks=True,
          ylabel="gsyn inhibitory (mV)",
          data_labels=[sEMD.label], yticks=True, xlim=(0, run_time)),
    title="Simple synfire chain example",
    annotations="Simulated with {}".format(p.name())
)
plt.show()

# end
p.end()
