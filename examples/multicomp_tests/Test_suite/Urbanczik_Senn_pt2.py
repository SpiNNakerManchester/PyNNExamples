import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np

# This test replicates the Urbanczik-Senn experiment in the 2014 paper.
# It runs the simulation on SpiNNaker, then computes the expected values in floating point on Host side and compares
# them by plotting both.


def test(g_D=2, g_L=0.1, exc_times=[1, 2, 5, 6], inh_times=[3, 4, 5, 6], exc_r_diff=[1, 1.3, 3.3, 1.5] ,
         Ee=4.667, Ei=-0.333):

    # Run for 22 ms
    runtime = 22000

    p.setup(timestep=1)

    # Dendritic initial weight
    weight_to_spike = 0.2

    # Somatic weight
    som_weight = 0.2
    learning_rate = 0.07


    # Shape generator for the excitatory input current
    xsin = np.linspace(0, np.pi, 50)
    xpar = np.linspace(-28, 27, 55)
    par = [0.0001754 * (i ** 2) for i in xpar]
    par.insert(28, 0.0)
    sin = [np.sin(i) for i in xsin]

    exc_som_val = [par[-1]]
    exc_som_val.extend(sin[3:47])
    exc_som_val.extend(par[:-1])


    dend_input = []
    dend_times = []

    # Generating dendritic inputs to be sent to SpiNNaker
    # 100 sources set to fire at every timestep. 100 ms period only one source per timestep sends 2.5, the others send 0
    for i in range(100):
        if i == 98:
            dend_input.append([2.5, 0])
            dend_times.append([98, 99])
        elif i == 99:
            dend_input.append([0, 2.5])
            dend_times.append([0, 99])
        else:
            dend_input.append([2.5, 0, 0])
            dend_times.append([i, i+1, 99])


    # Generating dendritic input for the Host side to check correctness of SpiNNaker execution
    exp_dend_input = []

    for i in range(len(dend_input)):
        tmp_input = []
        z = 0
        j = 0
        for k in range(runtime):
            if z in dend_times[i]:
                if k == 0 and dend_input[i][j] < 0:
                    tmp_input.append(0)
                else:
                    tmp_input.append(dend_input[i][j])
                j = (j + 1) % len(dend_times[i])
            else:
                tmp_input.append(0)
            z = (z + 1) % 100
        exp_dend_input.append(tmp_input)

    weights_val = [weight_to_spike for _ in range(len(dend_input))]


    # Generating somatic input for the Host side to check correctness of SpiNNaker execution
    exc_vals = [2.5 for i in range(runtime)]
    soma_vals = [exc_som_val[i % len(exc_som_val)] if i >= 1000 and i < 20000 else 0 for i in range(runtime)]
    soma_inh_vals = [2 if i >= 1000 and i < 20000 else 0 for i in range(runtime)]

    # Somatic ecitatory values to be sent to SpiNNaker
    exc_som_val_to_spinnaker = np.array(exc_som_val) / som_weight

    # Urbanczik-Senn population
    population = p.Population(1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, rate_update_threshold=0), label='population_1')
    input = []

    # Dendritic Inputs
    for i in range(100):
        input.append(p.Population(1, p.RateSourceArray(rate_times=dend_times[i], rate_values=dend_input[i], looping=1), label='exc_input_'+str(i)))

    # Somatic Inputs
    input2 = p.Population(1, p.RateSourceArray(rate_times=[_ for _ in range(1000, 1100)], rate_values=exc_som_val_to_spinnaker, looping=2), label='soma_exc_input')
    input3 = p.Population(1, p.RateSourceArray(rate_times=[1000, 20000], rate_values=[2/som_weight, 0]), label='soma_inh_input')

    plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10, w_max=10,
                                                                                     learning_rate=learning_rate),
        weight=weight_to_spike)

    # Dendritic Connections
    for i in range(100):
        p.Projection(input[i], population, p.OneToOneConnector(), synapse_type=plasticity,
                     receptor_type="dendrite_exc")

    # Somatic Connections
    p.Projection(input2, population, p.OneToOneConnector(), p.StaticSynapse(weight=som_weight),
                 receptor_type="soma_exc")
    p.Projection(input3, population, p.OneToOneConnector(), p.StaticSynapse(weight=som_weight),
                 receptor_type="soma_inh")

    population.record(['v', 'gsyn_exc', 'gsyn_inh'])

    # Run simulation
    p.run(runtime)

    # Extract recorded data
    u = population.get_data('v')
    v = population.get_data('gsyn_exc')
    rate = population.get_data('gsyn_inh')

    # Idnd = v.segments[0].filter(name='gsyn_exc')[0]

    # weights = [Idnd[i]/exc_vals[0] for i in range(len(Idnd))]


    # Plot values from SpiNNaker
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

    # End simulation
    p.end()

    # Save values from SpiNNaker in list form, to be plotted later for check against host values
    # u_vals = u.segments[0].filter(name='v')[0]
    # v_vals = v.segments[0].filter(name='gsyn_exc')[0]
    # um_vals = rate.segments[0].filter(name='gsyn_inh')[0]
    # x = [_ for _ in range(runtime)]

    # Parameters for Host side simulation
    # ge = 0
    # gi = 0
    # Um = []
    # V = []
    # U = []
    # dend_weight = weight_to_spike
    # somatic_weight = som_weight
    # Vrate = 0
    # Urate = 0
    # incoming_rates = [0 for _ in range(len(exp_dend_input))]

    # Run the simulation Host side to compare values extracted from SpiNNaker. Ignore this.
    # for i in range(runtime):
    #
    #     dend_curr = 0
    #
    #     for z in range(len(exp_dend_input)):
    #
    #         weights_val[z] += (learning_rate * incoming_rates[z] * (Urate - Vrate))
    #
    #         incoming_rates[z] = exp_dend_input[z][i]
    #
    #         dend_curr += incoming_rates[z] * weights_val[z]
    #
    #     V.append(dend_curr)
    #
    #     ge = soma_vals[i]
    #     gi = soma_inh_vals[i]
    #
    #     som_curr = (ge * Ee) + (gi * Ei)
    #
    #     gtot = g_D + g_L + (ge) + (gi)
    #
    #     som_voltage = float((dend_curr * g_D) + som_curr) / gtot
    #
    #     U.append(som_voltage)
    #
    #     Vrate = (dend_curr if (dend_curr > 0) else 0)
    #     Urate = (float((som_voltage if (som_voltage > 0) else 0) * (g_L + g_D)) / g_D)
    #
    #     if ge + gi != 0:
    #         Um.append(float((ge * Ee) + (gi * Ei)) / (ge + gi))
    #     else:
    #         Um.append(0)
    #
    # Um.insert(0, 0)
    # Um.pop()
    #
    # U.insert(0, 0)
    # U.pop()
    #
    # V.insert(0, 0)
    # V.pop()

    # Save the values on files for other plotting
    # with open("/localhome/g90604lp/um.txt", "w") as fp:
    #     for U in Um:
    #         fp.write(str(float(U)) + "\n")
    #
    # with open("/localhome/g90604lp/u.txt", "w") as fp:
    #     for U in u_vals:
    #         fp.writelines(str(float(U)) + "\n")
    #
    # with open("/localhome/g90604lp/v.txt", "w") as fp:
    #     for V in v_vals:
    #         fp.writelines(str(float(V)) + "\n")

    # Plot the values to compare the execution on SpiNNaker with the one on Host
    # plt.plot(x, Um, "--", color="red", linewidth=2, label="U_m")
    # plt.plot(x, u_vals, color="blue", linewidth=1.5, label="U")
    # plt.plot(x, v_vals, color="green", linewidth=1.5, label="V")
    # plt.plot(x, U, "--", color="aqua", linewidth=1.5, label="U expected")
    # plt.plot(x, V, "--", color="lightgreen", linewidth=1.5, label="V expected")
    # plt.grid(True)
    # plt.title("Urbanczik-Senn plasticity Voltages")
    # plt.xticks(x)
    # plt.legend()
    # plt.show()


    return False


def success_desc():
    return "Urbanczik-Senn test PASSED"


def failure_desc():
    return "Urbanczik-Senn test FAILED"


if __name__ == "__main__":
    Desc = test()
