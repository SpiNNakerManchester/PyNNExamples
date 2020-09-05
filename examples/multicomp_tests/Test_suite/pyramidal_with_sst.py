import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt


def test(apical_learning_rate=0.5, basal_learning_rate=0.25, sst_learning_rate=0.25,
         g_A=0.8, g_B=1, g_L=0.1, g_D=1.0, gsom=0.8,
         top_down_times=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
         bottom_up_times=[1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14],
         top_down_rates=[1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 1.5, 2, 2, 2, 2],
         bottom_up_rates=[4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 4.5, 2, 2, 2, 2]):

    runtime = 20

    p.setup(timestep=1)

    exc_apical_weight = 0.75
    inh_apical_weight = 1
    basal_weight = 1
    sst_dend_weight = 1
    sst_soma_weight = 0.5

    top_down = p.Population(1, p.RateSourceArray(rate_times=top_down_times, rate_values=top_down_rates, looping=4),
                            label='top_down_input')
    bottom_up = p.Population(1, p.RateSourceArray(rate_times=bottom_up_times, rate_values=bottom_up_rates, looping=4),
                             label='bottom_up_input')

    pyramidal = p.Population(1, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L), label='pyramidal')
    interneuron = p.Population(1, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D), label='interneuron')

    # basal_plasticity = p.STDPMechanism(
    #     timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
    #     weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-10, w_max=10,
    #                                                                learning_rates=(0, basal_learning_rate,
    #                                                                                apical_learning_rate, 0)),
    #     weight=basal_weight)

    apical_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-10, w_max=10,
                                                                   learning_rates=(0, 0, apical_learning_rate, 0)),
        weight=inh_apical_weight)

    sst_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10, w_max=10,
                                                                                     learning_rate=sst_learning_rate),
        weight=sst_dend_weight)

    p.Projection(top_down, pyramidal, p.OneToOneConnector(), p.StaticSynapse(weight=exc_apical_weight),
                 receptor_type="apical_exc")
    p.Projection(top_down, interneuron, p.OneToOneConnector(), p.StaticSynapse(weight=sst_soma_weight),
                 receptor_type='soma_exc')

    p.Projection(interneuron, pyramidal, p.OneToOneConnector(), synapse_type=apical_plasticity,
                 receptor_type="apical_inh")
    p.Projection(bottom_up, pyramidal, p.OneToOneConnector(), synapse_type=p.StaticSynapse(weight=basal_weight),
                 receptor_type="basal_exc")

    p.Projection(pyramidal, interneuron, p.OneToOneConnector(), synapse_type=sst_plasticity,
                 receptor_type='dendrite_exc')

    pyramidal.record(['v', 'gsyn_exc', 'gsyn_inh'])
    interneuron.record(['v', 'gsyn_exc'])

    p.run(runtime)

    u = pyramidal.get_data('v').segments[0].filter(name='v')[0]

    va = pyramidal.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    vb = pyramidal.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    usst = interneuron.get_data('v').segments[0].filter(name='v')[0]

    vsst = interneuron.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    # Plot values from SpiNNaker
    Figure(
        # membrane potential of the postsynaptic neuron
        Panel(u, ylabel="Soma Membrane potential (mV)",
              data_labels=[pyramidal.label], yticks=True, xlim=(0, runtime)),
        Panel(va, ylabel="Apical dendritic potential",
              data_labels=[pyramidal.label], yticks=True, xlim=(0, runtime)),
        Panel(vb, ylabel="Basal dendritic potential",
              data_labels=[pyramidal.label], yticks=True, xlim=(0, runtime)),
        Panel(usst, ylabel="SST somatic potential",
              data_labels=[interneuron.label], yticks=True, xlim=(0, runtime)),
        Panel(vsst, ylabel="SST dendritic potential",
              data_labels=[interneuron.label], yticks=True, xlim=(0, runtime)),
        title="Pyramidal + SST neurons",
        annotations="Simulated with {}".format(p.name())
    )

    plt.grid(True)

    plt.show()

    p.end()

    return True


def success_desc():
    return "Pyramidal model and sst neuron test PASSED"


def failure_desc():
    return "Pyramidal model and sst neuron test FAILED"


if __name__ == "__main__":

    if test():
        print success_desc()
    else:
        print failure_desc()
