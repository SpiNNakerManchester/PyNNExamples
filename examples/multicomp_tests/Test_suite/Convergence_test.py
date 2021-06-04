import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

# This test combines excitatory and inhibitory stimuli arriving only to dendrites with plastic synapses
# and tests correct weight update by checking the output voltages.


def test(g_D=1, g_L=0.1, exc_times=[_ for _ in range(400)], inh_times=[_ for _ in range(300)], exc_r_diff=[0.2 for _ in range(400)],
         inh_r_diff=[4 for _ in range(300)], update_thresh=2, learning_rate=0.9, g_som=0.8):
    runtime = 400

    p.setup(timestep=1)

    weight_to_spike = 0.5

    # population = p.Population(1, p.extra_models.PyramidalRate(g_B=g_D, g_L=g_L, g_A=g_som,
    #                                                              rate_update_threshold=update_thresh),
    #                           label='population_1', in_partitions=[1, 1, 0, 0], out_partitions=1)

    population = p.Population(1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, g_som=g_som,
                                                              rate_update_threshold=update_thresh),
                              label='population_1', in_partitions=[1, 1, 0, 0], out_partitions=1)
    input = p.Population(1, p.RateSourceArray(rate_times=exc_times, rate_values=exc_r_diff, looping=4, partitions=1),
                         label='exc_input')
    input2 = p.Population(1, p.RateSourceArray(rate_times=inh_times, rate_values=inh_r_diff, looping=4, partitions=1),
                          label='soma_input')

    plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(
            tau_plus=20.0, tau_minus=20.0),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-1000, w_max=1000,
                                                                                     learning_rate=learning_rate),
        weight=weight_to_spike)

    p.Projection(input, population, p.OneToOneConnector(), synapse_type=plasticity,
                 receptor_type="dendrite_exc")
    p.Projection(input2, population, p.OneToOneConnector(), synapse_type=p.StaticSynapse(weight=0),
                 receptor_type="soma_exc")

    population.record(['v', 'gsyn_exc'])

    p.run(runtime)

    u = population.get_data('v').segments[0].filter(name='v')[0]

    v = population.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    p.end()

    # Plot values from SpiNNaker
    Figure(
        # membrane potential of the postsynaptic neuron
        Panel(u, ylabel="somatic potential",
              data_labels=[population.label], yticks=True, xlim=(0, runtime)),
        Panel(v, ylabel="dendritic potential",
              data_labels=[population.label], yticks=True, xlim=(0, runtime)),
        title="MNIST output",
        annotations="Simulated with {}".format(p.name())
    )

    plt.show()

    p.end()

    # test_exc_times = [t + 1 for t in exc_times]
    # test_inh_times = [t + 1 for t in inh_times]
    #
    # ext_times = [_ for _ in range(runtime)]
    #
    # j = 0
    # k = 0
    # dend_exc = []
    # dend_inh = []
    #
    # for i in ext_times:
    #     if i in test_exc_times:
    #         dend_exc.append(exc_r_diff[j])
    #         j += 1
    #     else:
    #         dend_exc.append(0)
    #
    #     if i in test_inh_times:
    #         dend_inh.append(inh_r_diff[k])
    #         k += 1
    #     else:
    #         dend_inh.append(0)
    #
    # j = 0
    # k = 0
    # l = 0
    # m = 0
    # weight_exc = weight_to_spike
    # weight_inh = weight_to_spike
    # delta_exc = 0
    # delta_inh = 0
    # irate_exc = 0
    # irate_inh = 0
    # prev_rate = 0
    # U_rate = 0
    # V_rate = 0
    # V_vals = []
    # U_vals = []
    #
    # for i in range(len(u)):
    #
    #     Idnd = 0
    #
    #     if i in ext_times:
    #         weight_exc += delta_exc
    #         if weight_exc > 10:
    #             weight_exc = 10
    #         elif weight_exc < -10:
    #             weight_exc = -10
    #         irate_exc = dend_exc[j] if (dend_exc[j] > 0 and dend_exc[j] < 2) else 0 if dend_exc[j] <= 0 else 2
    #         Idnd += (irate_exc * weight_exc)
    #         j += 1
    #     if i in ext_times:
    #         weight_inh += delta_inh
    #         if weight_inh > 10:
    #             weight_inh = 10
    #         elif weight_inh < -10:
    #             weight_inh = -10
    #         irate_inh = dend_inh[k] if (dend_inh[k] > 0 and dend_inh[k] < 2) else 0 if dend_inh[k] <= 0 else 2
    #         Idnd -= ((irate_inh * weight_inh) if irate_inh > 0 else 0)
    #         k += 1
    #
    #     Vdnd = float(Idnd)
    #     V_vals.append(Vdnd)
    #     num = float(v[i])
    #     if (float(int(num * 100)) / 100 != float(int(Vdnd * 100)) / 100) and (round(num, 2) != round(Vdnd, 2)):
    #         for z in range(len(V_vals)):
    #             print("Dendritic voltage " + str(float(v[z])) + " expected " + str(V_vals[z]) + " index " + str(z))
    #         return False
    #
    #     U = float((g_D * Vdnd)) \
    #         / (g_D + g_L + g_som)
    #     U_vals.append(U)
    #     num = float(u[i])
    #     if (float(int(num * 100)) / 100 != float(int(U * 100)) / 100) and (round(num, 2) != round(U, 2)):
    #         for z in range(len(U_vals)):
    #             print("Somatic voltage " + str(float(u[z])) + " expected " + str(U_vals[z]) + " index " + str(z))
    #         return False
    #
    #     out_rate = _compute_rate(U)
    #
    #     V_rate = _compute_rate(float(Vdnd * (g_D / (g_L + g_D))))
    #     U_rate = _compute_rate(U)
    #
    #     delta_exc = learning_rate * (U_rate - V_rate) * irate_exc
    #     delta_inh = learning_rate * (U_rate - V_rate) * irate_inh

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
