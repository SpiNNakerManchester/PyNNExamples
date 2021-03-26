import spynnaker8 as p
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

# This test provides an oscillatory teaching current with the dendrite trying to follow it


def test(g_D=2, g_L=0.1, exc_times=[1, 2, 5, 6], inh_times=[3, 4, 5, 6], exc_r_diff=[1, 1.3, 3.3, 1.5], g_som=0.8):

    runtime = 3500

    p.setup(timestep=1)

    weight_to_spike = 0.2
    som_weight = 0
    learning_rate = 0.07

    # exc_som_val = [0.25, 0.25, 0, 0.25, 0.25, 0, 0, 0, 0, -0.25, -0.25, 0, -0.25, -0.25, 0, 0, 0, 0, 0, 0]

    # 25 0.04, 20 zeroes, 25 -0.04, 30 zeroes
    exc_som_val = []
    val = 0
    for i in range(25):
        exc_som_val.append(val + 0.0625)
    for i in range(20):
        exc_som_val.append(val)
    for i in range(25):
        exc_som_val.append(val - 0.0625)
    for i in range(30):
        exc_som_val.append(val)

    exc_t = [_ for _ in range(runtime)]
    #exc_vals = [2.5 if i == 0 else 0.5 if i % 2 == 0 else -0.5 for i in range(runtime)]
    exc_vals = [2.5 for i in range(runtime)]
    soma_vals = [exc_som_val[i % len(exc_som_val)] for i in range(runtime)]
    soma_inh_vals = [-2 for i in range(runtime)]

    population = p.Population(1, p.extra_models.IFExpRateTwoComp(g_D=g_D, g_L=g_L, g_som=g_som,
                                                                 rate_update_threshold=0), label='population_1',
                                                                 in_partitions=[1, 1, 0, 0], out_partitions=1)
    input = p.Population(1, p.RateSourceArray(rate_times=[0], rate_values=[2.5], partitions=1), label='exc_input')
    input2 = p.Population(1, p.RateSourceArray(rate_times=[_ for _ in range(100)], rate_values=exc_som_val, looping=1, partitions=1),
                          label='soma_exc_input')
    input3 = p.Population(1, p.RateSourceArray(rate_times=[0], rate_values=[-2], partitions=1), label='soma_exc_input')

    plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10, w_max=10,
                                                                                     learning_rate=learning_rate),
        weight=weight_to_spike)

    p.Projection(input, population, p.OneToOneConnector(), synapse_type=plasticity,
                 receptor_type="dendrite_exc")
    p.Projection(input2, population, p.OneToOneConnector(), p.StaticSynapse(weight=som_weight),
                 receptor_type="soma_exc")
    p.Projection(input3, population, p.OneToOneConnector(), p.StaticSynapse(weight=som_weight),
                 receptor_type="soma_exc")

    population.record(['v', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u = population.get_data('v')
    v = population.get_data('gsyn_exc')
    rate = population.get_data('gsyn_inh')

    p.end()

    u_vals = u.segments[0].filter(name='v')[0]
    v_vals = v.segments[0].filter(name='gsyn_exc')[0]
    rate_vals = rate.segments[0].filter(name='gsyn_inh')[0]


    V = []
    U = []
    exp_weights = []
    dend_weight = weight_to_spike
    somatic_weight = som_weight
    Vrate = 0
    Urate = 0
    incoming_rate = 0

    for i in range(len(soma_vals)):

        dend_weight += (learning_rate * incoming_rate * (Urate - Vrate))

        exp_weights.append(dend_weight)

        incoming_rate = exc_vals[i] if (exc_vals[i] > 0 and exc_vals[i] < 2) else 0 if exc_vals[i] <= 0 else 2

        dend_curr = incoming_rate * dend_weight
        V.append(dend_curr)

        ge = soma_vals[i]
        gi = soma_inh_vals[i]

        som_curr = (ge * g_som) + (gi * g_som)

        gtot = g_D + g_L + g_som

        som_voltage = float((dend_curr * g_D) + som_curr) / gtot

        U.append(som_voltage)

        Vrate = _compute_rate(dend_curr)
        Urate = _compute_rate((float(g_L + g_D) * som_voltage) / g_D)

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


    return True

def _compute_rate(voltage):

    tmp_rate = voltage if (voltage > 0 and voltage < 2) else 0 if voltage <= 0 else 2

    return tmp_rate

def success_desc():
    return "Dendritic prediction of oscillating somatic voltage test adapted (microcircuit enabled) PASSED"

def failure_desc():
    return "Dendritic prediction of oscillating somatic voltage test adapted (microcircuit enabled) FAILED"


if __name__ == "__main__":
    if test():
        print(success_desc())
    else:
        print(failure_desc())
