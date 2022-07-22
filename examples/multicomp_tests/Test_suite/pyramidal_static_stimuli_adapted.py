import spynnaker8 as p
import time

# Static test for the pyramidal model. Combined inputs arrive at the basal and apical dendrites with different weights
# and in different combinations. Correct behaviour for both excitatory and inhibitory apical synapses is checked.


def test(g_A=0.8, g_B=1, g_L=0.1, exc_times=[1, 2, 7, 8, 9, 10, 13, 14], inh_times=[3, 4, 7, 8, 11, 12, 13, 14],
         basal_times=[5, 6, 9, 10, 11, 12, 13, 14], exc_rates=[1, 1.3, -3.3, 1.5, 1.5, -3.3, 1.5, 2.5],
         inh_rates=[1, 1.3, -1.5, 3.3, 1.3, -2, 5, 3], basal_rates=[1, 1.3, -3.3, 1.5, -2, 1.3, 1.5, 2.5]):

    runtime = 21

    p.setup(timestep=1)

    exc_apical_weight = 1.5
    inh_apical_weight = 0.75
    basal_weight = 1.125

    # LP: Adjust n_syn_types to remove the unused ones!
    population = p.Population(
        1, p.extra_models.PyramidalRate(
            g_A=g_A, g_B=g_B, g_L=g_L),
        label='population_1', in_partitions=[1, 1, 1, 0], out_partitions=1)

    input1 = p.Population(1, p.RateSourceArray(rate_times=exc_times, rate_values=exc_rates, looping=4, partitions=1), label='apical_exc_input')
    input2 = p.Population(1, p.RateSourceArray(rate_times=inh_times, rate_values=inh_rates, looping=4, partitions=1), label='apical_inh_input')
    input3 = p.Population(1, p.RateSourceArray(rate_times=basal_times, rate_values=basal_rates, looping=4, partitions=1), label='basal_exc_input')

    p.Projection(input1, population, p.OneToOneConnector(), p.StaticSynapse(weight=exc_apical_weight),
                 receptor_type="apical_exc")
    # Testing static inhibition on apical dendrite
    p.Projection(input2, population, p.OneToOneConnector(), p.StaticSynapse(weight=inh_apical_weight),
                 receptor_type="apical_inh")
    p.Projection(input3, population, p.OneToOneConnector(), p.StaticSynapse(weight=basal_weight),
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

    U_pyral = []

    for i in range(runtime):

        Iapical = 0
        Ibasal = 0

        if i in exc_times:
            exc = exc_rates[j] if (exc_rates[j] > 0 and exc_rates[j] < 2) else 0 if exc_rates[j] <= 0 else 2
            Iapical += (exc * exc_apical_weight)
            j += 1
        if i in inh_times:
            inh = inh_rates[k] if (inh_rates[k] > 0 and inh_rates[k] < 2) else 0 if inh_rates[k] <= 0 else 2
            Iapical += (inh * inh_apical_weight)
            k += 1
        if i in basal_times:
            bas = basal_rates[l] if (basal_rates[l] > 0 and basal_rates[l] < 2) else 0 if basal_rates[l] <= 0 else 2
            Ibasal += (bas * basal_weight)
            l += 1

        Va.append(Iapical)
        Vb.append(Ibasal)
        U.append(float(g_B * Ibasal + g_A * Iapical)/(g_L + g_B + g_A))

        
        # These checks are for comparison with PyraLNet from Laura, to verify that the
        # Solution to the differential equations they used is the same as ours.
        # To verify it compare U_pyral with U
        som = U_pyral[i-1] if i >= 3 else U[i-1] if i < 3 else 0
        ub = Vb[i-1] if i > 0 else 0
        ua = Va[i-1] if i > 0 else 0

        du = (-g_L * som + g_B * (Vb[i-1] - som) + g_A * (Va[i-1] - som))

        U_pyral.append(som + du)


    U.insert(0, 0)
    U.pop()

    Va.insert(0, 0)
    Va.pop()

    Vb.insert(0, 0)
    Vb.pop()

    print(u)
    print(U)
    print(U_pyral)

    for i in range(runtime):

        num = float(va[i])
        if (float(int(num * 1000)) / 1000 != float(int(Va[i] * 1000)) / 1000) and (round(num, 3) != round(Va[i], 3)):
            print("Apical voltage " + str(float(va[i])) + " != " + str(Va[i]) + " t " + str(i))
            return False

        num = float(vb[i])
        if (float(int(num * 1000)) / 1000 != float(int(Vb[i] * 1000)) / 1000) and (round(num, 3) != round(Vb[i], 3)):
            print("Basal voltage " + str(float(vb[i])) + " != " + str(Vb[i]) + " t " + str(i))
            return False

        num = float(u[i])
        if (float(int(num * 1000)) / 1000 != float(int(U[i] * 1000)) / 1000) and (round(num, 3) != round(U[i], 3)):
            print("Somatic voltage " + str(float(u[i])) + " != " + str(U[i]) + " t " + str(i))
            return False

    return True


def success_desc():
    return "Pyramidal static stimuli test adapted (microcircuit enabled) PASSED"


def failure_desc():
    return "Pyramidal static stimuli test adapted (microcircuit enabled) FAILED"


if __name__ == "__main__":

    if test():
        print(success_desc())
    else:
        print(failure_desc())