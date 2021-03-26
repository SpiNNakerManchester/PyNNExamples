import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

# Test used for the EOY report. Two dendritic sources active intermittently.
# Simplified stimuli similar to US experiment.


def test(g_D=2, g_L=0.1, exc_times=[1, 2, 5, 6], inh_times=[3, 4, 5, 6], exc_r_diff=[1, 1.3, 3.3, 1.5] ,
         Ee=4.667, Ei=-0.333, graphic=False):

    runtime = 2102

    p.setup(timestep=1)

    weight_to_spike = 0.7
    learning_rate = 0.3

    exc_t = [_ for _ in range(runtime)]

    dend_exc = [1.5 if (i % 4 == 0) else 0 for i in range(runtime)]

    dend_inh = dend_exc[:]
    dend_inh.pop(-1)
    dend_inh.pop(-1)
    dend_inh.insert(0, 0)
    dend_inh.insert(0, 0)

    soma_exc = [1 if (i % 4 == 0 and i < runtime / 2) else 0 for i in range(runtime)]

    soma_inh = soma_exc[:]
    soma_inh.pop(-1)
    soma_inh.pop(-1)
    soma_inh.insert(0, 0)
    soma_inh.insert(0, 0)

    population = p.Population(1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, rate_update_threshold=0, e_rev_E=Ee, e_rev_I=Ei),
                              label='population_1')
    dend_exc_input = p.Population(1, p.RateSourceArray(rate_times=exc_t, rate_values=dend_exc), label='exc_input')
    dend_exc_input2 = p.Population(1, p.RateSourceArray(rate_times=exc_t, rate_values=dend_inh), label='exc_input2')
    input2 = p.Population(1, p.RateSourceArray(rate_times=exc_t[:runtime], rate_values=soma_exc), label='soma_exc_input')
    input3 = p.Population(1, p.RateSourceArray(rate_times=exc_t[:runtime], rate_values=soma_inh), label='soma_inh_input')

    plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10, w_max=10,
                                                                                     learning_rate=learning_rate),
        weight=weight_to_spike)

    p.Projection(dend_exc_input, population, p.OneToOneConnector(), synapse_type=plasticity,
                 receptor_type="dendrite_exc")
    p.Projection(dend_exc_input2, population, p.OneToOneConnector(), synapse_type=plasticity,
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

    if graphic:
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

    Um = []
    V = []
    U = []
    dend_weight_e = weight_to_spike
    dend_weight_i = weight_to_spike
    somatic_weight = weight_to_spike
    Vrate = 0
    Urate = 0
    incoming_rate_e = 0
    incoming_rate_i = 0

    for i in range(runtime):

        dend_weight_e += (learning_rate * incoming_rate_e * (Urate - Vrate))
        dend_weight_i += (learning_rate * incoming_rate_i * (Urate - Vrate))

        incoming_rate_e = dend_exc[i]
        incoming_rate_i = dend_inh[i]

        dend_curr = (incoming_rate_e * dend_weight_e) + (incoming_rate_i * dend_weight_i)
        V.append(dend_curr)

        ge = soma_exc[i]
        gi = soma_inh[i]

        som_curr = (somatic_weight * ge * Ee) + (somatic_weight * gi * Ei)

        gtot = g_D + g_L + (ge * somatic_weight) + (gi * somatic_weight)

        som_voltage = float((dend_curr * g_D) + som_curr) / gtot

        U.append(som_voltage)

        Vrate = dend_curr if dend_curr > 0 else 0
        Urate = (float(som_voltage * (g_L + g_D)) / g_D) if som_voltage > 0 else 0

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

    for i in range(runtime):

        num = float(v_vals[i])
        if (float(int(num * 10)) / 10 != float(int(V[i] * 10)) / 10) and (
                round(num, 1) != round(V[i], 1)):
            print("Dendritic voltage " + str(float(v_vals[i])) + " expected " + str(V[i]) + " index " + str(i))
            return False

        num = float(u_vals[i])
        if (float(int(num * 10)) / 10 != float(int(U[i] * 10)) / 10) and (
                round(num, 1) != round(U[i], 1)):
            print("Somatic voltage " + str(float(u_vals[i])) + " expected " + str(U[i]) + " index " + str(i))
            return False

    if graphic:
        plt.plot(x, Um, "--", color="red", linewidth=2, label="U_m")
        plt.plot(x, u_vals, color="blue", linewidth=1.5, label="U")
        plt.plot(x, v_vals, color="green", linewidth=1.5, label="V")
        plt.plot(x, U, "--", color="aqua", linewidth=1.5, label="U expected")
        plt.plot(x, V, "--", color="lightgreen", linewidth=1.5, label="V expected")
        plt.grid(True)
        plt.title("Urbanczik-Senn plasticity Voltages")
        plt.xticks(x)
        plt.legend()
        plt.show()

    return True


def success_desc():
    return "2 sources simple sin test PASSED"


def failure_desc():
    return "2 sources simple sin test FAILED"


if __name__ == "__main__":

    if test():
        print(success_desc())
    else:
        print(failure_desc())
