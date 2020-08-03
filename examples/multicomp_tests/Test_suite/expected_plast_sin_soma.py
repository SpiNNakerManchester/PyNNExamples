import math
import matplotlib.pyplot as plt

# This test provides a fixed teaching current with the dendrite trying to follow it


def test(g_D=2, g_L=0.1, exc_times=[1, 2, 5, 6], inh_times=[3, 4, 5, 6], exc_r_diff=[1, 1.3, 3.3, 1.5] ,
         Ee=4.667, Ei=-0.333):

    runtime = 500

    weight_to_spike = 0.7
    som_weight = 0.7
    learning_rate = 0.3

    # exc_som_val = [0.25, 0.25, 0, 0.25, 0.25, 0, 0, 0, 0, -0.25, -0.25, 0, -0.25, -0.25, 0, 0, 0, 0, 0, 0]

    # 25 0.04, 20 zeroes, 25 -0.04, 30 zeroes
    exc_som_val = []
    for i in range(25):
        exc_som_val.append(0.04)
    for i in range(20):
        exc_som_val.append(0)
    for i in range(25):
        exc_som_val.append(-0.04)
    for i in range(35):
        exc_som_val.append(0)

    exc_t = [_ for _ in range(runtime)]
    exc_vals = [2.5 if i == 0 else 0 for i in range(runtime)]
    #exc_vals = [1.5 if i == 0 else 0 for i in range(runtime)]
    soma_vals = [exc_som_val[i % len(exc_som_val)] for i in range(runtime)]
    #soma_vals = [1 if i == 0 else -1 if i == 36 else 0 for i in range(runtime)]
    soma_inh_vals = [2 if (i == 0) else 0 for i in range(runtime)]


    x = [_ for _ in range(runtime)]

    ge = 0
    gi = 0
    Um = []
    V = []
    U = []
    exp_weights = []
    dend_weight = weight_to_spike
    somatic_weight = som_weight
    Vrate = 0
    Urate = 0
    incoming_rate = 0
    phi_max = 150
    k = 0.5
    beta = 5
    delta = 1

    for i in range(len(soma_vals)):

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


        if dend_weight > 0:
            Vrate = float(phi_max) / (1 + k * math.exp(beta * (delta - dend_curr)))
        else:
            Vrate = 0

        if som_voltage > 0:
            Urate = float(phi_max) / (1 + k * math.exp(beta * (delta - (float(som_voltage * (g_L + g_D)) / g_D))))
        else:
            Urate = 0

        #Urate = float(som_voltage * (g_L + g_D)) / g_D
        #Vrate = dend_curr

        if ge + gi != 0:
            Um.append(float((ge * Ee) + (gi * Ei)) / (ge + gi))
        else:
            Um.append(0)

    plt.plot(x, Um, "--", color="red", linewidth=2, label="U_m")
    plt.plot(x, U, "--", color="blue", linewidth=1.5, label="U expected")
    plt.plot(x, V, "--", color="green", linewidth=1.5, label="V expected")
    plt.grid(True)
    plt.title("Urbanczik-Senn plasticity Voltages")
    plt.xticks(x)
    plt.legend()
    plt.show()

    plt.plot(x, exp_weights, "--",  color="green", linewidth=2, label="weights")
    plt.grid(True)
    plt.title("Urbanczik-Senn plasticity dendritic weight")
    plt.xticks(x)
    plt.legend()
    plt.show()



    return True

if __name__ == "__main__":
    test()
