import spynnaker8 as p

# Plasticity check on the pyramidal model. In this test only the basal dendrite is plastic, the apical one is static and
# it receives both excitatory and inhibitory signals. The test can be performend on any number of postsynaptic neurons


def test(learning_rate=0.25 , g_A=0.8, g_B=1, g_L=0.1, exc_times=[1, 2, 7, 8, 9, 10, 13, 14], inh_times=[3, 4, 7, 8, 11, 12, 13, 14],
         basal_times=[5, 6, 9, 10, 11, 12, 13, 14], exc_rates=[1, -1.3, 3.3, 1.5, 1.5, 3.3, 1.5, -2.5],
         inh_rates=[-1, 1.3, 1.5, 3.3, -1.3, 2, 5, 3], basal_rates=[1, 1.3, 3.3, 1.5, -2, 1.3, 1.5, 2.5]):

    runtime = 20

    p.setup(timestep=1)

    exc_apical_weight = 1.25
    inh_apical_weight = 0.75
    basal_weight = 0.5

    post_neurons = 200

    population = p.Population(
        post_neurons, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L),
        label='population_1', in_partitions=[4, 6, 4, 0], out_partitions=1)
    input1 = p.Population(1, p.RateSourceArray(rate_times=exc_times, rate_values=exc_rates, looping=4, partitions=1),
                         label='apical_exc_input')
    input2 = p.Population(1, p.RateSourceArray(rate_times=inh_times, rate_values=inh_rates, looping=4, partitions=1),
                         label='apical_inh_input')
    input3 = p.Population(1, p.RateSourceArray(rate_times=basal_times, rate_values=basal_rates, looping=4, partitions=1),
                         label='basal_exc_input')

    plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10, w_max=10,
                                                                                     learning_rate=learning_rate),
        weight=basal_weight)

    p.Projection(input1, population, p.AllToAllConnector(), p.StaticSynapse(weight=exc_apical_weight),
                 receptor_type="apical_exc")
    #static inhibition on apical dendrite
    p.Projection(input2, population, p.AllToAllConnector(), p.StaticSynapse(weight=inh_apical_weight),
                 receptor_type="apical_inh")
    p.Projection(input3, population, p.AllToAllConnector(), synapse_type=plasticity,
                 receptor_type="basal_exc")

    population.record(['v', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u = population.get_data('v').segments[0].filter(name='v')[0]

    va = population.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    vb = population.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    p.end()

    j = 0
    k = 0
    l = 0

    Va = []
    Vb = []
    U = []

    Urate = 0
    Vb_rate = 0

    plastic_basal_weight = basal_weight

    incoming_rate = 0

    exc_test_rates = []
    inh_test_rates = []
    basal_test_rates = []

    ext_times = [_ for _ in range(runtime)]

    for i in ext_times:
        if i in exc_times:
            exc_test_rates.append(exc_rates[j])
            j += 1
        else:
            exc_test_rates.append(0)

        if i in inh_times:
            inh_test_rates.append(inh_rates[k])
            k += 1
        else:
            inh_test_rates.append(0)

        if i in basal_times:
            basal_test_rates.append(basal_rates[l])
            l += 1
        else:
            basal_test_rates.append(0)

    j = 0
    k = 0
    l = 0

    for i in range(runtime):

        Iapical = 0
        Ibasal = 0

        if i in ext_times:

            exc = exc_test_rates[j] if (exc_test_rates[j] > 0 and exc_test_rates[j] < 2) else 0 if exc_test_rates[j] <= 0 else 2
            Iapical += (exc * exc_apical_weight)
            j += 1

            inh = inh_test_rates[k] if (inh_test_rates[k] > 0 and inh_test_rates[k] < 2) else 0 if inh_test_rates[k] <= 0 else 2
            Iapical += (inh * inh_apical_weight)
            k += 1


            plastic_basal_weight += (learning_rate * (Urate - Vb_rate) * incoming_rate)

            if plastic_basal_weight > 10:
                plastic_basal_weight = 10
            elif plastic_basal_weight < -10:
                plastic_basal_weight = -10

            bas = basal_test_rates[l] if (basal_test_rates[l] > 0 and basal_test_rates[l] < 2) else 0 if basal_test_rates[l] <= 0 else 2
            incoming_rate = bas
            Ibasal += (bas * plastic_basal_weight)
            l += 1

        som_voltage = float(g_B * Ibasal + g_A * Iapical)/(g_L + g_B + g_A)

        Va.append(Iapical)
        Vb.append(Ibasal)
        U.append(som_voltage)

        Vb_rate = (Ibasal if (Ibasal > 0 and Ibasal < 2) else 0 if Ibasal <= 0 else 2)
        som_voltage = float(som_voltage * (g_L + g_B + g_A)) / g_B
        Urate = (som_voltage if (som_voltage > 0 and som_voltage < 2) else 0 if som_voltage <= 0 else 2)


    U.insert(0, 0)
    U.pop()

    Va.insert(0, 0)
    Va.pop()

    Vb.insert(0, 0)
    Vb.pop()

    for n in range(post_neurons):
        for i in range(runtime):

            num = float(va[i][n])
            if (float(int(num * 100)) / 100 != float(int(Va[i] * 100)) / 100) and (round(num, 2) != round(Va[i], 2)):
                print("neuron " + str(n))
                for j in range(i + 1):
                    print("Apical voltage " + str(float(va[j][n])) + " != " + str(Va[j]) + " time " + str(j))
                return False

            num = float(vb[i][n])
            if (float(int(num * 100)) / 100 != float(int(Vb[i] * 100)) / 100) and (round(num, 2) != round(Vb[i], 2)):
                print("neuron " + str(n))
                for j in range(i + 1):
                    print("Basal voltage " + str(float(vb[j][n])) + " != " + str(Vb[j]) + " time " + str(j))
                return False

            num = float(u[i][n])
            if (float(int(num * 100)) / 100 != float(int(U[i] * 100)) / 100) and (round(num, 2) != round(U[i], 2)):
                print("neuron " + str(n))
                for j in range(i + 1):
                    print("Somatic voltage " + str(float(u[j][n])) + " != " + str(U[j]) + " time " + str(j))
                return False

    return True


def success_desc():
    return "Pyramidal basal plasticity test adapted (microcircuit enabled) PASSED"


def failure_desc():
    return "Pyramidal basal plasticity test adapted (microcircuit enabled) FAILED"


if __name__ == "__main__":

    if test():
        print(success_desc())
    else:
        print(failure_desc())
