import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

# This test provides a fixed teaching current with the dendrite trying to follow it


def test(g_D=2, g_L=0.1, exc_times=[1, 2, 5, 6], inh_times=[3, 4, 5, 6], exc_r_diff=[1, 1.3, 3.3, 1.5] ,
         Ee=4.667, Ei=-0.333):

    runtime = 15

    p.setup(timestep=1)

    weight_to_spike = 0.7
    learning_rate = 0.3

    exc_t = [_ for _ in range(runtime)]
    exc_vals = [1.5 if i == 0 else 0 for i in range(runtime)]
    soma_vals = [1, 0, 0, 0, 0, -1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    soma_inh_vals = [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, -1, 0, 0, 0, 0]

    population = p.Population(1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, rate_update_threshold=0, e_rev_E=Ee, e_rev_I=Ei),
                              label='population_1')
    input = p.Population(1, p.RateSourceArray(rate_times=exc_t, rate_values=exc_vals), label='exc_input')
    input2 = p.Population(1, p.RateSourceArray(rate_times=exc_t, rate_values=soma_vals), label='soma_exc_input')
    input3 = p.Population(1, p.RateSourceArray(rate_times=exc_t, rate_values=soma_inh_vals), label='soma_inh_input')

    plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10, w_max=10,
                                                                                     learning_rate=learning_rate),
        weight=weight_to_spike)

    p.Projection(input, population, p.OneToOneConnector(), synapse_type=plasticity,
                 receptor_type="dendrite_exc")
    p.Projection(input2, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="soma_exc")
    p.Projection(input3, population, p.OneToOneConnector(), p.StaticSynapse(weight=weight_to_spike),
                 receptor_type="soma_inh")

    population.record(['v', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u = population.get_data('v')
    v = population.get_data('gsyn_exc')
    rate = population.get_data('gsyn_inh')

    Idnd = v.segments[0].filter(name='gsyn_exc')[0]

    weights = [Idnd[i]/exc_vals[0] for i in range(len(Idnd))]


    Figure(
        # membrane potential of the postsynaptic neuron
        Panel(u.segments[0].filter(name='v')[0],
              ylabel="Soma Membrane potential (mV)",
              data_labels=[population.label], yticks=True, xlim=(0, runtime)),
        Panel(v.segments[0].filter(name='gsyn_exc')[0],
              ylabel="Dendrite membrane potential",
              data_labels=[population.label], yticks=True, xlim=(0, runtime)),
        Panel(rate.segments[0].filter(name='gsyn_inh')[0],
              ylabel="Rate",
              data_labels=[population.label], yticks=True, xlim=(0, runtime)),
        title="multicompartment example",
        annotations="Simulated with {}".format(p.name())
    )

    plt.grid(True)

    plt.show()

    p.end()

    u_vals = u.segments[0].filter(name='v')[0]
    v_vals = v.segments[0].filter(name='gsyn_exc')[0]
    rate_vals = rate.segments[0].filter(name='gsyn_inh')[0]
    x = [_ for _ in range(runtime)]

    ge = 0
    gi = 0
    Um = []
    V = []
    U = []
    dend_weight = weight_to_spike
    somatic_weight = weight_to_spike
    Vrate = 0
    Urate = 0
    incoming_rate = 0
    exp_weights = []

    for i in range(len(soma_inh_vals)):

        dend_weight += (learning_rate * incoming_rate * (Urate - Vrate))

        exp_weights.append(dend_weight)

        incoming_rate += exc_vals[i]

        dend_curr = incoming_rate * dend_weight
        V.append(dend_curr)

        ge += soma_vals[i]
        gi += soma_inh_vals[i]

        som_curr = (somatic_weight * ge * Ee) + (somatic_weight * gi * Ei)

        gtot = g_D + g_L + (ge * somatic_weight) + (gi * somatic_weight)

        som_voltage = float((dend_curr * g_D) + som_curr) / gtot

        U.append(som_voltage)

        Vrate = dend_curr
        Urate = (float(som_voltage * (g_L + g_D)) / g_D)

        if ge + gi != 0:
            Um.append(float(ge * Ee + gi * Ei) / (ge + gi))
        else:
            Um.append(0)

    Um.insert(0, 0)
    Um.pop()

    U.insert(0, 0)
    U.pop()

    V.insert(0, 0)
    V.pop()

    exp_weights.insert(0, 0)
    exp_weights.pop()

    plt.plot(x, Um, "--", color="red", linewidth=2, label="U_m")
    plt.plot(x, u_vals, color="blue", linewidth=1.5, label="U")
    plt.plot(x, v_vals, color="green", linewidth=1.5, label="V")
    #plt.plot(x, U, "--", color="aqua", linewidth=1.5, label="U expected")
    #plt.plot(x, V, "--", color="lightgreen", linewidth=1.5, label="V expected")
    plt.grid(True)
    plt.title("Urbanczik-Senn plasticity Voltages")
    plt.xticks(x, fontsize=16)
    plt.yticks(fontsize=16)
    plt.legend()
    plt.show()

    plt.plot(x, weights, color="blue", linewidth=2, label="weights")
    plt.plot(x, exp_weights, "--", color="green", linewidth=2, label="weights")
    plt.grid(True)
    plt.title("Urbanczik-Senn plasticity dendritic weight")
    plt.xticks(x)
    plt.legend()
    plt.show()



    return True

def success_desc():
    return "Dendritic prediction of fixed somatic voltage test PASSED"

def failure_desc():
    return "Dendritic prediction of fixed somatic voltage test FAILED"


if __name__ == "__main__":
    test()
