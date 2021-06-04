import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

# This test combines excitatory and inhibitory stimuli arriving only to dendrites with plastic synapses
# and tests correct weight update by checking the output voltages.


def test(g_D=1, g_L=0.1, exc_times=[_ for _ in range(400)], inh_times=[_ for _ in range(4000)], exc_r_diff=[0.2 for _ in range(400)],
         inh_r_diff=[4 for _ in range(4000)], update_thresh=2, learning_rate=0.2, g_som=0.8):
    runtime = 4000

    p.setup(timestep=1)

    weight_to_spike = 0.5
    apical_weight = 0.25
    basal_weight = 0.75
    inh_weight = 1
    apical_rate = 0.5
    upper_rate = 0.05

    pyramidal = p.Population(1, p.extra_models.PyramidalRate(g_B=g_D, g_L=g_L, g_A=g_som,
                                                                 rate_update_threshold=update_thresh),
                              label='pyramidal', in_partitions=[1, 1, 1, 0], out_partitions=1)

    interneuron = p.Population(1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, g_som=g_som,
                                                              rate_update_threshold=update_thresh),
                              label='interneuron', in_partitions=[1, 1, 0, 0], out_partitions=1)

    top_down = p.Population(1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, g_som=g_som,
                                                               rate_update_threshold=update_thresh, teach=False),
                               label='top_down', in_partitions=[0, 1, 0, 0], out_partitions=1)

    input = p.Population(1, p.RateSourceArray(rate_times=inh_times, rate_values=inh_r_diff, looping=4, partitions=1),
                         label='exc_input')

    lateral_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-1000, w_max=1000,
                                                                                     learning_rate=learning_rate),
        weight=weight_to_spike)

    upper_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-1000, w_max=1000,
                                                                                     learning_rate=upper_rate),
        weight=basal_weight)

    basal_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(learning_rate, 0, 0, 0)),
        weight=basal_weight)

    apical_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-1000, w_max=1000,
                                                                   learning_rates=(apical_rate, 0, 0, 0)),
        weight=inh_weight)

    p.Projection(input, pyramidal, p.OneToOneConnector(), synapse_type=basal_plasticity,
                 receptor_type="basal_exc")
    p.Projection(pyramidal, interneuron, p.OneToOneConnector(), synapse_type=lateral_plasticity,
                 receptor_type="dendrite_exc")
    p.Projection(interneuron, pyramidal, p.OneToOneConnector(), synapse_type=apical_plasticity,
                 receptor_type="apical_inh")
    p.Projection(pyramidal, top_down, p.OneToOneConnector(), synapse_type=upper_plasticity,
                 receptor_type="dendrite_exc")
    p.Projection(top_down, interneuron, p.OneToOneConnector(), synapse_type=p.StaticSynapse(weight=0),
                 receptor_type="soma_exc")
    p.Projection(top_down, pyramidal, p.OneToOneConnector(), synapse_type=p.StaticSynapse(weight=apical_weight),
                 receptor_type="apical_exc")

    pyramidal.record(['v', 'gsyn_exc', 'gsyn_inh'])
    interneuron.record(['v', 'gsyn_exc', 'gsyn_inh'])
    top_down.record(['v', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u = pyramidal.get_data('v').segments[0].filter(name='v')[0]

    va = pyramidal.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    vb = pyramidal.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    usst = interneuron.get_data('v').segments[0].filter(name='v')[0]

    vsst = interneuron.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    utop = top_down.get_data('v').segments[0].filter(name='v')[0]

    vbtop = top_down.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    p.end()

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

def _compute_rate(voltage):

    tmp_rate = voltage if (voltage > 0 and voltage < 2) else 0 if voltage <= 0 else 2

    return tmp_rate

def success_desc():
    return "Dendritc plasticity test adapted (microcircuit enabled) PASSED"

def failure_desc():
    return "Dendritic plasticity test adapted (microcircuit enabled) FAILED"


if __name__ == "__main__":

    if test():
        print(success_desc())
    else:
        print(failure_desc())
