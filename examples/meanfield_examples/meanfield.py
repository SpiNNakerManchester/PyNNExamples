#import pyNN.spiNNaker as sim
import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

#from spynnaker.pyNN.models.neuron.builds.meanfield_base import MeanfieldBase

runtime = 500

time_step = 1.0

n_neurons = 10

p.setup(time_step)
#sim.set_numbre_of_neurons_per_core(sim.MeanfieldBase, 1)

pop = p.Population(1, p.extra_models.Meanfield())
pop.record('Ve')

p.run(runtime)

#A_pop = pop.get_data('a')
Ve_pop = pop.get_data('Ve')
#v = neo.segments[0].filter(name='Ve')[0]
#print(v)

#runtime =500
#p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)
#nNeurons=1
#p.set_number_of_neurons_per_core(p.Meanfield, nNeurons)
print(Ve_pop.segments[0].filter(name='Ve')[0])

#Figure(
#    Panel(Ve_pop.segments[0].filter(name='Ve')[0],
#          ylabel="MF (mV)",
#          yticks=True,
#          xlim=(0, runtime),
#          xticks=True),
#    title="test"
#)
#plt.show()

p.end()
