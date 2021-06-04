import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import random

# Simulation of the network described in the Supplementary Material of the Sacramento paper (NIPS 2018). 
# The parameters come from section S1

def test(pyramidal_inter_learning_rate = 0.0002375, inter_pyramidal_lerning_rate=0.0005,
            g_A=0.8, g_B=1, g_L=0.1, g_D=1.0, gsom=0.8):

    runtime = 1050

    p.setup(timestep=1)

    exc_apical_weight = random.uniform(-1, 1)
    inh_apical_weight = random.uniform(-1, 1)
    basal_weight = random.uniform(-1, 1)
    sst_dend_weight = random.uniform(-1, 1)
    sst_soma_weight = 0
    top_down_basal_weight = random.uniform(-1, 1)

    input_pop_size = 30
    pyramidal_pop_size = 20
    top_down_pop_size = 10

    values = [0.2 if i < 333 else 0.4 if i >= 333 and i < 666 else 0.6 for i in range(1000)]
    rate_vals = []

    for _ in range(105):
        rate_vals.extend(values)

    input_population = [p.Population(1, p.RateSourceArray(rate_times=[_ for _ in range(len(rate_vals))], rate_values=rate_vals, looping=4,
                                                    partitions=1), label='bottom_up_input') for _ in range(input_pop_size)]

    pyramidal = p.Population(pyramidal_pop_size, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L),
                                label='pyramidal', in_partitions=[1, 1, 1, 0], out_partitions=1)
    interneuron = p.Population(top_down_pop_size, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D),
                                label='interneuron', in_partitions=[1, 1, 0, 0], out_partitions=1)
    # Slightly better results using a two-comp as top-down pop
    top_down = p.Population(top_down_pop_size, p.extra_models.IFExpRateTwoComp(g_L=g_L, g_D=g_D, teach=False),
                                label='top_down_population', in_partitions=[0, 1, 0, 0], out_partitions=1)


    apical_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(inter_pyramidal_lerning_rate, 0,
                                                                                   0, 0)),
        weight=inh_apical_weight)

    sst_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-1000, w_max=1000,
                                                                                     learning_rate=pyramidal_inter_learning_rate),
        weight=sst_dend_weight)


    p.Projection(top_down, pyramidal, p.AllToAllConnector(random_weight_matrix=True), p.StaticSynapse(weight=exc_apical_weight),
                 receptor_type="apical_exc")
    p.Projection(top_down, interneuron, p.OneToOneConnector(random_weight_matrix=True), p.StaticSynapse(weight=sst_soma_weight),
                 receptor_type='soma_exc')

    p.Projection(interneuron, pyramidal, p.AllToAllConnector(random_weight_matrix=True), synapse_type=apical_plasticity,
                 receptor_type="apical_inh")
    p.Projection(pyramidal, interneuron, p.AllToAllConnector(random_weight_matrix=True), synapse_type=sst_plasticity,
                 receptor_type='dendrite_exc')

    p.Projection(pyramidal, top_down, p.AllToAllConnector(random_weight_matrix=True), p.StaticSynapse(weight=top_down_basal_weight),
                 receptor_type='dendrite_exc')

    for i in range(input_pop_size):
        p.Projection(input_population[i], pyramidal, p.AllToAllConnector(random_weight_matrix=True), p.StaticSynapse(weight=random.uniform(-1,1)),
            receptor_type="basal_exc")
    

    pyramidal.record(['v', 'gsyn_exc', 'gsyn_inh'])
    interneuron.record(['v', 'gsyn_exc', 'synapse'])
    top_down.record(['v', 'gsyn_inh'])

    p.run(runtime)

    u = pyramidal.get_data('v').segments[0].filter(name='v')[0]

    va = pyramidal.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    vb = pyramidal.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    usst = interneuron.get_data('v').segments[0].filter(name='v')[0]

    vsst = interneuron.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    utop = top_down.get_data('v').segments[0].filter(name='v')[0]

    vbtop = top_down.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    weights = interneuron.get_data('synapse')

    # Plot values from SpiNNaker
    Figure(
        # membrane potential of the postsynaptic neuron
        Panel(utop, ylabel="Top-down somatic potential",
              data_labels=[top_down.label], yticks=True, xlim=(0, runtime)),
        Panel(usst, ylabel="SST somatic potential",
              data_labels=[interneuron.label], yticks=True, xlim=(0, runtime)),
        Panel(vbtop, ylabel="Top-down dendritic potential",
              data_labels=[top_down.label], yticks=True, xlim=(0, runtime)),
        Panel(vsst, ylabel="SST dendritic potential",
              data_labels=[interneuron.label], yticks=True, xlim=(0, runtime)),
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
              data_labels=[top_down.label], yticks=True, xlim=(0, runtime)),
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
