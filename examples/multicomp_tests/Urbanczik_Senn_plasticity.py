import spynnaker8 as p
import time
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

def Urbanczik_Senn_plasticity():

    runtime = 40
    nNeurons = 1
    source_pops = 1

    p.setup(timestep=1.0, min_delay=1.0, max_delay=144.0)

    dendritic_weight = 0.2
    somatic_weight = 0.2

    dend_rate = 10

    g_I = [2 if i == 0 else 0 for i in range(runtime)]
    g_E = [1 if (i > 2 and i % 2 != 0) else -1 for i in range(runtime)]
    dend = [-1 if i % 2 != 0 else 1 for i in range(2, runtime)]
    dend_input = [10]
    dend_input.extend(dend)

    population1 = p.Population(nNeurons, p.extra_models.IFExpRateTwoComp(starting_rate=0), label='population_1')

    g_E_source = p.Population(1, p.RateSourceArray(rate_times=range(runtime), rate_values=g_E))
    g_I_source = p.Population(1, p.RateSourceArray(rate_times=range(runtime), rate_values=g_I))

    # Input populations with rate 10 Hz
    dend_source = [p.Population(1, p.RateSourceArray(rate_times=range(1, runtime), rate_values=dend_input)) for _ in range(source_pops)]

    p.Projection(g_E_source, population1, p.OneToOneConnector(), synapse_type=p.StaticSynapse(weight=somatic_weight),
                 receptor_type="soma_exc")
    p.Projection(g_I_source, population1, p.OneToOneConnector(), synapse_type=p.StaticSynapse(weight=somatic_weight),
                 receptor_type="soma_inh")

    plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=0, w_max=20,
                                                                                     learning_rate=0.07),
        weight=dendritic_weight)

    dend_projs = [p.Projection(dend_source[i], population1, p.OneToOneConnector(),
                               synapse_type=plasticity, receptor_type="dendrite_exc") for i in range(source_pops)]

    population1.record(['v', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u = population1.get_data('v')
    v = population1.get_data('gsyn_exc')
    rate = population1.get_data('gsyn_inh')

    Figure(
        # membrane potential of the postsynaptic neuron
        Panel(u.segments[0].filter(name='v')[0],
              ylabel="Soma Membrane potential (mV)",
              data_labels=[population1.label], yticks=True, xlim=(0, runtime)),
        Panel(v.segments[0].filter(name='gsyn_exc')[0],
              ylabel="Dendrite membrane potential",
              data_labels=[population1.label], yticks=True, xlim=(0, runtime)),
        Panel(rate.segments[0].filter(name='gsyn_inh')[0],
              ylabel="Rate",
              data_labels=[population1.label], yticks=True, xlim=(0, runtime)),
        title="multicompartment example",
        annotations="Simulated with {}".format(p.name())
    )

    plt.grid(True)

    plt.show()

    p.end()

    return False


if __name__ == "__main__":
    if  Urbanczik_Senn_plasticity():
        print "PASSED!!!"
    else:
        print "FAILED"