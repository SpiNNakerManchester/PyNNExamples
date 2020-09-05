import spynnaker8 as p

# Plasticity check on the pyramidal model. In this test both the basal and apical dendrites are plastic.
# The apical dendrite receives both excitatory and inhibitory signals


def test(apical_learning_rate = 0.5, basal_learning_rate=0.25 , g_A=0.8, g_B=1, g_L=0.1,
         exc_times=[1, 2, 7, 8, 9, 10, 13, 14], inh_times=[3, 4, 7, 8, 11, 12, 13, 14],
         basal_times=[5, 6, 9, 10, 11, 12, 13, 14], exc_rates=[1, 1.3, 3, 1.5, 1.5, 3.75, 1.5, 2.5],
         inh_rates=[1, 1.125, 1.5, 3, 1.5, 0.75, 2, 2.5], basal_rates=[1, 1.3, 3.75, 1.5, 2, 1.25, 1.5, 2.5]):

    runtime = 20

    p.setup(timestep=1)

    exc_apical_weight = 1.5
    inh_apical_weight = 1
    basal_weight = 1.5

    population = p.Population(1, p.extra_models.PyramidalRate(g_A=g_A, g_B=g_B, g_L=g_L), label='population_1')
    input = p.Population(1, p.RateSourceArray(rate_times=exc_times, rate_values=exc_rates, looping=4),
                         label='apical_exc_input')
    input2 = p.Population(1, p.RateSourceArray(rate_times=inh_times, rate_values=inh_rates, looping=4),
                          label='apical_inh_input')
    input3 = p.Population(1, p.RateSourceArray(rate_times=basal_times, rate_values=basal_rates, looping=4),
                          label='basal_exc_input')

    basal_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-10, w_max=10,
                                                                   learning_rates=(0, basal_learning_rate,
                                                                                  apical_learning_rate, 0)),
        weight=basal_weight)

    apical_plasticity = p.STDPMechanism(
        timing_dependence=p.extra_models.TimingDependenceMulticompBern(),
        weight_dependence=p.extra_models.WeightDependencePyramidal(w_min=-10, w_max=10,
                                                                   learning_rates=(0, basal_learning_rate,
                                                                                  apical_learning_rate, 0)),
        weight=inh_apical_weight)

    p.Projection(input, population, p.OneToOneConnector(), p.StaticSynapse(weight=exc_apical_weight),
                 receptor_type="apical_exc")
    p.Projection(input2, population, p.OneToOneConnector(), synapse_type=apical_plasticity,
                 receptor_type="apical_inh")
    p.Projection(input3, population, p.OneToOneConnector(), synapse_type=basal_plasticity,
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
    Va_plast = 0

    plastic_basal_weight = basal_weight
    plastic_apical_weight = inh_apical_weight

    incoming_rate = 0
    incoming_inh_rate = 0

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

            plastic_apical_weight += (apical_learning_rate * (0 - Va_plast) * incoming_inh_rate)

            if plastic_apical_weight > 10:
                plastic_apical_weight = 10
            elif plastic_apical_weight < -10:
                plastic_apical_weight = -10

            incoming_inh_rate = inh_test_rates[k]

            Iapical -= (inh_test_rates[k] * plastic_apical_weight)
            k += 1

            plastic_basal_weight += (basal_learning_rate * (Urate - Vb_rate) * incoming_rate)

            if plastic_basal_weight > 10:
                plastic_basal_weight = 10
            elif plastic_basal_weight < -10:
                plastic_basal_weight = -10

            incoming_rate = basal_test_rates[l]

            Ibasal += (basal_test_rates[l] * plastic_basal_weight)
            l += 1

        som_voltage = float(g_B * Ibasal + g_A * Iapical)/(g_L + g_B + g_A)

        Va.append(Iapical)
        Vb.append(Ibasal)
        U.append(som_voltage)

        Vb_rate = (Ibasal if (Ibasal > 0) else 0)
        Urate = (float((som_voltage if (som_voltage > 0) else 0) * (g_L + g_B + g_A)) / g_B)
        Va_plast = Iapical


    U.insert(0, 0)
    U.pop()

    Va.insert(0, 0)
    Va.pop()

    Vb.insert(0, 0)
    Vb.pop()

    for i in range(runtime):

        num = float(va[i])
        if (float(int(num * 100)) / 100 != float(int(Va[i] * 100)) / 100) and (round(num, 2) != round(Va[i], 2)):
            print "Apical voltage " + str(float(va[i])) + " != " + str(Va[i]) + " time " + str(i)
            return False

        num = float(vb[i])
        if (float(int(num * 100)) / 100 != float(int(Vb[i] * 100)) / 100) and (round(num, 2) != round(Vb[i], 2)):
            print "Basal voltage " + str(float(vb[i])) + " != " + str(Vb[i]) + " time " + str(i)
            return False

        num = float(u[i])
        if (float(int(num * 100)) / 100 != float(int(U[i] * 100)) / 100) and (round(num, 2) != round(U[i], 2)):
            print "Somatic voltage " + str(float(u[i])) + " != " + str(U[i]) + " time " + str(i)
            return False

    return True


def success_desc():
    return "Pyramidal full plasticity test PASSED"


def failure_desc():
    return "Pyramidal full plasticity test FAILED"


if __name__ == "__main__":

    if test():
        print success_desc()
    else:
        print failure_desc()
