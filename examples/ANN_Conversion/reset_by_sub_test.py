import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


p.setup(time_scale_factor=1, timestep=1)

pop1 = p.Population(1, p.SpikeSourceArray([10, 30, 50, 70, 90, 110]))
pop2 = p.Population(1, p.IF_curr_exp())

p.Projection(pop1, pop2, p.AllToAllConnector(), p.StaticSynapse(weight=4.0, delay=1.0))

pop2[0].record(['v'])

p.run(250)

v = pop2.get_data('v')

p.end()

Figure(
    # membrane potential of the postsynaptic neuron
    Panel(v.segments[0].filter(name='v')[0],
          ylabel="Membrane potential (mV)",
          data_labels=[pop2.label], yticks=True, xlim=(0, 250), xticks=True),
    annotations="Simulated with {}".format(p.name())
)

plt.show()