import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt
import numpy as np

# This test replicates the Urbanczik-Senn experiment from the 2014 paper, but using the somatic conductances described
# by the paper from Sacramento 2018.
# It runs the simulation on SpiNNaker, then computes the expected values in floating point on Host side and compares
# them by plotting both.


def test(g_D=2, g_L=0.1, exc_times=[1, 2, 5, 6], inh_times=[3, 4, 5, 6], exc_r_diff=[1, 1.3, 3.3, 1.5] ,
         Ee=4.667, Ei=-0.333, g_som=1.0, graphic=False):

    # Run for 22 ms
    runtime = 22000

    # Duration of teching inputs
    teaching_time = 20000

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
    soma_vals = [exc_som_val[i % len(exc_som_val)] if i >= 1000 and i < teaching_time else 0 for i in range(runtime)]
    soma_inh_vals = [2 if i >= 1000 and i < teaching_time else 0 for i in range(runtime)]
    U_m = [(soma_vals[i] * Ee + soma_inh_vals[i] * Ei) / (soma_vals[i] + soma_inh_vals[i]) 
                if soma_vals[i] + soma_inh_vals[i] != 0 else 0  for i in range(teaching_time)]


    # !-------------------------------------------------------------------------------!
    # Host side simulation

    # Parameters for Host side simulation
    V = []
    U = []
    Vrate = 0
    Urate = 0
    incoming_rates = [0 for _ in range(len(exp_dend_input))]

    rec_weights = [[] for _ in range(len(exp_dend_input))]

    # Run the simulation Host side to compare values extracted from SpiNNaker.
    for i in range(runtime):

        dend_curr = 0

        for z in range(len(exp_dend_input)):

            weights_val[z] += (learning_rate * incoming_rates[z] * (Urate - Vrate))

            rec_weights[z].append(weights_val[z])

            irate = exp_dend_input[z][i] if (exp_dend_input[z][i] > 0 and exp_dend_input[z][i] < 2) else 0 if exp_dend_input[z][i] <= 0 else 2
            
            incoming_rates[z] = irate
            
            dend_curr += irate * weights_val[z]

        V.append(dend_curr)

        if i == 0:
            som_curr = 0
        elif i < teaching_time:
            som_curr = (soma_vals[i] * Ee + soma_inh_vals[i] * Ei) / (soma_vals[i] + soma_inh_vals[i]) if soma_vals[i] + soma_inh_vals[i] != 0 else 0
        else:
            som_curr = ((g_D * g_som * V[i]) / (g_D + g_L))
            # This value is for SpiNNaker, to simulate the absence of teaching input, since it is not possible
            # to explicitly do it using the Sacramento formulation.
            U_m.append(som_curr)

        gtot = g_D + g_L + g_som

        som_voltage = float((dend_curr * g_D) + som_curr * g_som) / gtot

        U.append(som_voltage)

        Vrate = _compute_rate(dend_curr)
        Urate = _compute_rate(float((som_voltage if (som_voltage > 0) else 0) * (g_L + g_D)) / g_D)

    U.insert(0, 0)
    U.pop()

    V.insert(0, 0)
    V.pop()

    # !-------------------------------------------------------------------------------!
    # SpiNNaker side simulation

    # Urbanczik-Senn population
    population = p.Population(
        1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, rate_update_threshold=0, g_som=g_som),
        label='population_1', in_partitions=[1, 1, 0, 0], out_partitions=1)
    input = []

    # Dendritic Inputs
    for i in range(100):
        input.append(p.Population(1, p.RateSourceArray(rate_times=[i for i in range(runtime)], rate_values=exp_dend_input[i], looping=4,
            partitions=1), label='exc_input_'+str(i)))

    # Somatic Inputs
    input2 = p.Population(1, p.RateSourceArray(rate_times=[_ for _ in range(runtime)], rate_values=U_m, looping=4, partitions=1), label='soma_exc_input')

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

    population.record(['v', 'gsyn_exc', 'gsyn_inh'])

    # Run simulation
    p.run(runtime)

    # Extract recorded data
    u = population.get_data('v')
    v = population.get_data('gsyn_exc')
    rate = population.get_data('gsyn_inh')


    # Plot values from SpiNNaker
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

    # End simulation
    p.end()

    # Save values from SpiNNaker in list form, to be plotted later for check against host values
    u_vals = u.segments[0].filter(name='v')[0]
    v_vals = v.segments[0].filter(name='gsyn_exc')[0]
    um_vals = rate.segments[0].filter(name='gsyn_inh')[0]

    # !-------------------------------------------------------------------------------!
    # Results checking

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

    x = [ _ for _ in range(runtime)]
    
    if graphic:
        plt.plot(x, U_m, "--", color="red", linewidth=2, label="U_m")
        plt.plot(x, u_vals, color="blue", linewidth=1.5, label="U")
        plt.plot(x, v_vals, color="green", linewidth=1.5, label="V")
        #plt.plot(x, U, "--", color="aqua", linewidth=1.5, label="U expected")
        #plt.plot(x, V, "--", color="lightgreen", linewidth=1.5, label="V expected")
        plt.grid(True)
        plt.title("Urbanczik-Senn plasticity Voltages")
        plt.xticks(x)
        plt.legend()
        plt.show()

    return True

def _compute_rate(voltage):

    tmp_rate = voltage if (voltage > 0 and voltage < 2) else 0 if voltage <= 0 else 2

    return tmp_rate


def success_desc():
    return "Urbanczik-Senn test adapted (microcircuit enabled) PASSED"


def failure_desc():
    return "Urbanczik-Senn test adapted (microcircuit enabled) FAILED"


if __name__ == "__main__":
    if test(graphic=True):
        print(success_desc())
    else:
        print(failure_desc())