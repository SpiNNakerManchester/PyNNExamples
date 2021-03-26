import spynnaker8 as p

# Plasticity check on the pyramidal model. In this test only the basal dendrite is plastic, the apical one is static and
# it receives both excitatory and inhibitory signals.


def test(learning_rate=0.25 , g_A=0.8, g_B=1, g_L=0.1, exc_times=[1, 2, 7, 8, 9, 10, 13, 14], inh_times=[3, 4, 7, 8, 11, 12, 13, 14],
         basal_times=[5, 6, 9, 10, 11, 12, 13, 14], exc_rates=[1, 1.3, 3.3, 1.5, 1.5, 3.3, 1.5, 2.5],
         inh_rates=[1, 1.3, 1.5, 3.3, 1.3, 2, 5, 3], basal_rates=[1, 1.3, 3.3, 1.5, 2, 1.3, 1.5, 2.5]):

    runtime = 20

    p.setup(timestep=1)

    exc_apical_weight = 1.5
    inh_apical_weight = 0.75
    basal_weight = 1.125

    population = p.Population(1, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L), label='population_1')
    input = p.Population(1, p.RateSourceArray(rate_times=exc_times, rate_values=exc_rates, looping=4),
                         label='apical_exc_input')
    input2 = p.Population(1, p.RateSourceArray(rate_times=inh_times, rate_values=inh_rates, looping=4),
                          label='apical_inh_input')
    input3 = p.Population(1, p.RateSourceArray(rate_times=basal_times, rate_values=basal_rates, looping=4),
                          label='basal_exc_input')

    plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependenceMultiplicativeMulticompBern(w_min=-10, w_max=10,
                                                                                     learning_rate=learning_rate),
        weight=basal_weight)

    p.Projection(input, population, p.OneToOneConnector(), p.StaticSynapse(weight=exc_apical_weight),
                 receptor_type="apical_exc")
    #static inhibition on apical dendrite
    p.Projection(input2, population, p.OneToOneConnector(), p.StaticSynapse(weight=inh_apical_weight),
                 receptor_type="apical_inh")
    p.Projection(input3, population, p.OneToOneConnector(), synapse_type=plasticity,
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
            Iapical += (exc_test_rates[j] * exc_apical_weight)
            j += 1

            Iapical -= (inh_test_rates[k] * inh_apical_weight)
            k += 1


            plastic_basal_weight += (learning_rate * (Urate - Vb_rate) * incoming_rate)

            incoming_rate = basal_test_rates[l]

            Ibasal += (basal_test_rates[l] * plastic_basal_weight)
            l += 1

        som_voltage = float(g_B * Ibasal + g_A * Iapical)/(g_L + g_B + g_A)

        Va.append(Iapical)
        Vb.append(Ibasal)
        U.append(som_voltage)

        Vb_rate = (Ibasal if (Ibasal > 0) else 0)
        Urate = (float((som_voltage if (som_voltage > 0) else 0) * (g_L + g_B + g_A)) / g_B)


    U.insert(0, 0)
    U.pop()

    Va.insert(0, 0)
    Va.pop()

    Vb.insert(0, 0)
    Vb.pop()


    for i in range(runtime):

        num = float(va[i])
        if (float(int(num * 100)) / 100 != float(int(Va[i] * 100)) / 100) and (round(num, 2) != round(Va[i], 2)):
            print("Apical voltage " + str(float(va[i])) + " != " + str(Va[i]))
            return False

        num = float(vb[i])
        if (float(int(num * 100)) / 100 != float(int(Vb[i] * 100)) / 100) and (round(num, 2) != round(Vb[i], 2)):
            print("Basal voltage " + str(float(vb[i])) + " != " + str(Vb[i]))
            return False

        num = float(u[i])
        if (float(int(num * 100)) / 100 != float(int(U[i] * 100)) / 100) and (round(num, 2) != round(U[i], 2)):
            print("Somatic voltage " + str(float(u[i])) + " != " + str(U[i]))
            return False

    return True


def success_desc():
    return "Pyramidal basal plasticity test PASSED"


def failure_desc():
    return "Pyramidal basal plasticity test FAILED"


if __name__ == "__main__":

    if test():
        print(success_desc())
    else:
        print(failure_desc())
