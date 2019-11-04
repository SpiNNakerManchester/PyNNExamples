import math
import matplotlib.pyplot as plt


def _expand(runtime, set, delay, weight, scale):

    set = [(set[i] + delay) * scale for i in range(len(set))]
    expanded_set = []
    j = 0
    for i in range(runtime):
        if len(set) == 0 or j >= len(set):
            expanded_set.append(0)
        elif i != set[j]:
            expanded_set.append(0)
        else:
            tmp_weight = weight
            while j < len(set) - 1 and set[j+1] == i:
                j += 1
                tmp_weight += weight
            expanded_set.append(tmp_weight)
            j += 1

    return expanded_set


def compute_rate(U):

    # parameters
    phi_max = 500
    k = 0.5
    beta = 5
    delta = 1

    current_rate = float(phi_max) / (1 + k * math.exp(beta * (delta - (U))))

    return current_rate


def golden_potentials(runtime_ms, timesteps_per_ms, somatic_exc, somatic_inh, dendritic_exc, dendritic_inh, delay, weight):

    # parameters
    # coupling conductance
    gsd = 2
    # leaky conductance
    gl = 0.1
    # synaptic conductances (soma)
    ge = 0
    gi = 0

    Isyn_dnd = 0
    # reversal potentials
    Ee = 4.667
    Ei = 0.333

    # Return values
    V = list()
    U = list()
    rate = list()
    out_spikes = list()

    V_prev = 0
    U_prev = 0

    # Get the number of timesteps
    runtime = runtime_ms * timesteps_per_ms

    somatic_spikes_exc = _expand(runtime, somatic_exc, delay, weight, timesteps_per_ms)
    somatic_spikes_inh = _expand(runtime, somatic_inh, delay, weight, timesteps_per_ms)
    dendritic_spikes_exc = _expand(runtime, dendritic_exc, delay, weight, timesteps_per_ms)
    dendritic_spikes_inh = _expand(runtime, dendritic_inh, delay, weight, timesteps_per_ms)

    tau_syn_soma = 5
    tau_syn_dend = 5

    timestep_duration_microsec = float(1000.0) / timesteps_per_ms
    timestep_duration_millisec = float(timestep_duration_microsec) / 1000

    # Exponential decay factors of the synapses
    # decay = e^-(1/tau_syn_soma)
    decay_soma = math.exp(-timestep_duration_millisec / tau_syn_soma)
    init_soma = (float(tau_syn_soma) / timestep_duration_millisec) * (1.0 - decay_soma)
    decay_dend = math.exp(-timestep_duration_millisec / tau_syn_dend)
    init_dend = (float(tau_syn_dend) / timestep_duration_millisec) * (1.0 - decay_dend)

    mean_isi_ticks = 65000
    time_to_spike = 65000
    time_since_last_spike = 0

    for i in range(runtime):

        ge += init_soma * somatic_spikes_exc[i]
        gi += init_soma * somatic_spikes_inh[i]

        g_tot = gl + gsd
        R_tot = 1 / g_tot

        # IS THIS CORRECT?
        Isyn_dnd += init_dend * dendritic_spikes_exc[i] - init_dend * dendritic_spikes_inh[i]

        # CHECK THIS, BECAUSE IN C IT HAS BEEN SET AS A DIFFERENCE, BUT IN THE PAPER IS A SUM AND Ei IS NEGATIVE!
        Isyn_soma = (ge * (Ee - U_prev)) - (gi * (Ei - U_prev))

        # Dendritic potential
        V.append(Isyn_dnd + math.exp(float(-gl * timestep_duration_microsec)/1000) * (V_prev - Isyn_dnd))

        alpha = float((gsd * V[i] + Isyn_soma)) / g_tot


        # Somatic potential
        U.append(alpha + math.exp(float(-g_tot * timestep_duration_microsec) / 1000) * (U_prev - alpha))

        U_prev = U[i]
        V_prev = V[i]

        ge *= decay_soma
        gi *= decay_soma

        Isyn_dnd *= decay_dend

        rate.append(compute_rate(U[i]))

        mean_isi_ticks = (timesteps_per_ms * 1000) / rate[i]

        time_to_spike -= 1
        time_since_last_spike += 1

        time_to_spike = mean_isi_ticks - time_since_last_spike

        if time_to_spike <= 0:
            out_spikes.append(i)
            time_since_last_spike = 0

    return V, U, rate, out_spikes


if __name__ == "__main__":

    # This values are the same as the ones in the SpiNNaker test.
    # Note that after you receive a spike, the conductances doesn't go to 0, but starts to decay
    # "compensate for the valve behaviour of a synapse in biology (spike goes in, synapse opens, then closes slowly)
    #  and the leaky behaviour of the neuron" <- SpiNNaker C code

    # Length of the simulation in ms
    runtime = 50

    # Number of timesteps in one ms
    timesteps_per_ms = 10

    # Lists of spikes per receptor
    somatic_spikes_exc = [1, 2, 3, 4, 5, 6]
    somatic_spikes_inh = []
    dendritic_spikes_exc = []
    dendritic_spikes_inh = []

    # Synaptic delay
    delay = 1
    # Synaptic weight
    weight = 1

    V, U, rate, out_spikes = golden_potentials(
        runtime, timesteps_per_ms, somatic_spikes_exc, somatic_spikes_inh,
        dendritic_spikes_exc, dendritic_spikes_inh, delay, weight)

    spikes_val = [1 for i in range(len(out_spikes))]

    runtime *= timesteps_per_ms

    plt.subplot(4, 1, 1)
    plt.xlim(-1, runtime + 1)
    plt.plot(out_spikes, spikes_val, "o", color="black")
    #plt.xlabel("time (ms)")
    plt.ylabel("Spikes")
    plt.grid(True)

    plt.subplot(4, 1, 2)
    plt.xlim(-1, runtime + 1)
    plt.plot(range(runtime), U, "-o", color="blue")
    # plt.xlabel("time (ms)")
    plt.ylabel("Somatic Potential")
    plt.grid(True)

    plt.subplot(4, 1, 3)
    plt.xlim(-1, runtime + 1)
    plt.plot(range(runtime), V, "-o", color="green")
    # plt.xlabel("time (ms)")
    plt.ylabel("Dendritic Potential")
    plt.grid(True)

    plt.subplot(4, 1, 4)
    plt.xlim(-1, runtime + 1)
    plt.plot(range(runtime), rate, "-o", color="red")
    plt.xlabel("time (ms)")
    plt.ylabel("Rate")
    plt.grid(True)

    plt.tight_layout()
    plt.show()

    print V
    print U

