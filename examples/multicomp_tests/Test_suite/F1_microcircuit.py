import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import random

# Simulation of the network described in the Supplementary Material of the Sacramento paper (NIPS 2018). 
# The parameters come from section S1

def test(pyramidal_inter_learning_rate = 0.0011875, inter_pyramidal_lerning_rate=0.0005, pyramidal_pyramidal_l0_leraning_rate=0.0011875,
            pyramidal_pyramidal_l1_leraning_rate=0.0005, g_A=0.8, g_B=1, g_L=0.1, g_D=1.0, gsom=0.8):

    runtime = 6000

    p.setup(timestep=1)

    exc_apical_weight = 1
    inh_apical_weight = 0.25
    basal_weight = 0.125
    sst_dend_weight = 0.5
    sst_soma_weight = 0
    top_down_basal_weight = 0.75

    input_pop_size = 30
    pyramidal_pop_size = 20
    top_down_pop_size = 10

    rate_vals = [(0.2) for _ in range(runtime)]

    input_population = [p.Population(1, p.RateSourceArray(rate_times=[_ for _ in range(len(rate_vals))], rate_values=rate_vals, looping=1,
                                                    partitions=1), label='bottom_up_input') for _ in range(input_pop_size)]

    teacher = [p.Population(1, p.RateSourceArray(rate_times=[_ for _ in range(runtime)], rate_values=[0 if i < 200 else 0.8 for i in range(runtime)], looping=1,
                                                    partitions=1), label='teacher') for _ in range(top_down_pop_size)]

    pyramidal = p.Population(pyramidal_pop_size, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L),
                                label='pyramidal', in_partitions=[1, 1, 1, 0], out_partitions=1)
    interneuron = p.Population(top_down_pop_size, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D),
                                label='interneuron', in_partitions=[1, 1, 0, 0], out_partitions=1)
    top_down = p.Population(top_down_pop_size, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D),
                                label='top_down_population', in_partitions=[1, 1, 0, 0], out_partitions=1)


    apical_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(inter_pyramidal_lerning_rate, 0,
                                                                                   0, 0)),
        weight=inh_apical_weight)

    basal_plasticity_l0 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(pyramidal_pyramidal_l0_leraning_rate, 0,
                                                                                   0, 0)),
        weight=basal_weight)

    basal_plasticity_l1 = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(pyramidal_pyramidal_l1_leraning_rate, 0,
                                                                                   0, 0)),
        weight=basal_weight)

    sst_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-1000, w_max=1000,
                                                                                     learning_rate=pyramidal_inter_learning_rate),
        weight=sst_dend_weight)


    p.Projection(top_down, pyramidal, p.AllToAllConnector(), p.StaticSynapse(weight=exc_apical_weight),
                 receptor_type="apical_exc")
    p.Projection(top_down, interneuron, p.OneToOneConnector(), p.StaticSynapse(weight=sst_soma_weight),
                 receptor_type='soma_exc')

    p.Projection(interneuron, pyramidal, p.AllToAllConnector(), synapse_type=apical_plasticity,
                 receptor_type="apical_inh")
    p.Projection(pyramidal, interneuron, p.AllToAllConnector(), synapse_type=sst_plasticity,
                 receptor_type='dendrite_exc')

    p.Projection(pyramidal, top_down, p.AllToAllConnector(), synapse_type=basal_plasticity_l1,
                 receptor_type='dendrite_exc')

    for i in range(input_pop_size):
        p.Projection(input_population[i], pyramidal, p.AllToAllConnector(), synapse_type=basal_plasticity_l0,
            receptor_type="basal_exc")

    for i in range(top_down_pop_size):
        p.Projection(teacher[i], top_down, p.OneToOneConnector(), p.StaticSynapse(weight=sst_soma_weight),
            receptor_type="soma_exc")
    

    pyramidal.record(['v', 'gsyn_exc', 'gsyn_inh'])
    interneuron.record(['v', 'gsyn_exc'])
    top_down.record(['v', 'gsyn_inh'])

    p.run(runtime)

    u = pyramidal.get_data('v').segments[0].filter(name='v')[0]

    va = pyramidal.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    vb = pyramidal.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    usst = interneuron.get_data('v').segments[0].filter(name='v')[0]

    vsst = interneuron.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    utop = top_down.get_data('v').segments[0].filter(name='v')[0]

    vbtop = top_down.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    # Plot values from SpiNNaker
    Figure(
        # membrane potential of the postsynaptic neuron
        Panel(utop, ylabel="Top-down somatic potential",
              data_labels=[interneuron.label], yticks=True, xlim=(0, runtime)),
        Panel(vbtop, ylabel="Top-down dendritic potential",
              data_labels=[interneuron.label], yticks=True, xlim=(0, runtime)),
        Panel(usst, ylabel="SST somatic potential",
              data_labels=[interneuron.label], yticks=True, xlim=(0, runtime)),
        Panel(vsst, ylabel="SST dendritic potential",
              data_labels=[top_down.label], yticks=True, xlim=(0, runtime)),
        title="Top-Down + SST neurons",
        annotations="Simulated with {}".format(p.name()),
    )

    plt.grid(True)

    plt.show()


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
        Panel(utop, ylabel="Top-down somatic potential",
              data_labels=[interneuron.label], yticks=True, xlim=(0, runtime)),
        title="Pyramidal + SST neurons",
        annotations="Simulated with {}".format(p.name())
    )

    plt.show()

    p.end()

    return True


def success_desc():
    return "S1 microcircuit test PASSED"


def failure_desc():
    return "S1 microcircuit test FAILED"


if __name__ == "__main__":

    if test():
        print(success_desc())
    else:
        print(failure_desc())
