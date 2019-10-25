import spynnaker8 as sim
# import statistics
import math
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


runtime = 1000
sim.setup(timestep=0.1)
nNeurons = 80  # number of neurons in each population


#rates = [400, 800, 1200]
rng = sim.NumpyRNG(seed=1)

# weights = sim.RandomDistribution('uniform', [1.0, 3.0], rng=rng)

# cell_params_lif = {'cm': 0.25,
#                    'i_offset': 0.0,
#                    'tau_m': 20.0,
#                    'tau_refrac': 2.0,
#                    'tau_syn_E': 5.0,
#                    'tau_syn_I': 5.0,
#                    'v_reset': -70.0,
#                    'v_rest': -65.0,
#                    'v_thresh': -50.0,
#                    'v': -55
#                    }

neuron_params = {'cm'        : 0.25,  # nF
                 'i_offset'  : 0.0,   # nA
                 'tau_m'     : 10.0,  # ms
                 'tau_refrac': 2.0,   # ms
                 'tau_syn_E' : 0.5,   # ms
                 'tau_syn_I' : 0.5,   # ms
                 'v_reset'   : -65.0, # mV
                 'v_rest'    : -65.0, # mV
                 'v_thresh'  : -50.0  # mV
                }


weight = 0.0878

rate = 23200
# rate = 12800

params = {"rate": rate, "poisson_weight": weight}

pop = sim.Population(nNeurons, sim.IF_curr_exp(**neuron_params), label='pop')

input = sim.Population(
    nNeurons, sim.PoissonSource, params, label='Poisson')

pop.add_poisson_source(input)



pop.record('all'
#             , indexes = [0]
           )

sim.run(runtime)


exc_data = pop.get_data('all')

I_poi = exc_data.segments[0].filter(name='gsyn_exc')[0].magnitude

print "mean current input: {}".format(I_poi.mean())

F = Figure(
    # plot data for postsynaptic neuron
    Panel(exc_data.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[pop.label], yticks=True, xlim=(0, runtime)
          ),
    Panel(exc_data.segments[0].filter(name='gsyn_exc')[0],
          ylabel="gsyn excitatory (mV)",
          data_labels=[pop.label], yticks=True, xlim=(0, runtime)
          ),
    Panel(exc_data.segments[0].filter(name='gsyn_inh')[0],
          ylabel="gsyn inhibitory (mV)",
          data_labels=[pop.label], yticks=True, xlim=(0, runtime)
          ),
    Panel(exc_data.segments[0].spiketrains,
          yticks=True, markersize=2, xlim=(0, runtime)
          ),
    )

plt.show()

sim.end()
print "job done"
