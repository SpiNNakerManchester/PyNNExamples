import spynnaker8 as p

# Plasticity check on the pyramidal model. In this test both the basal and apical dendrites are plastic.
# The apical dendrite receives both excitatory and inhibitory signals


def test(apical_learning_rate = 0.5, basal_learning_rate=0.25 , g_A=0.8, g_B=1, g_L=0.1,
         exc_times=[1, 2, 7, 8, 9, 10, 13, 14], inh_times=[3, 4, 7, 8, 11, 12, 13, 14],
         basal_times=[5, 6, 9, 10, 11, 12, 13, 14], exc_rates=[1, 1.3, 3, 1.5, 1.5, -3.75, 1.5, -2.5],
         inh_rates=[-1, 1.125, 1.5, 3, 1.5, -0.75, 2, 2.5], basal_rates=[1, -1.3, 3.75, 1.5, 2, 1.25, 1.5, 2.5]):

    runtime = 20

    p.setup(timestep=1)

    exc_apical_weight = 1.5
    inh_apical_weight = 1
    basal_weight = 1.5

    population1 = p.Population(
        40*8, p.extra_models.IFExpRateTwoComp(g_L=g_L), 
        label='population_1', in_partitions=[0, 0, 0, 0], out_partitions=4)

    population2 = p.Population(
        70*8, p.extra_models.IFExpRateTwoComp(g_L=g_L), 
        label='population_2', in_partitions=[0, 0, 0, 0], out_partitions=7)

    population3 = p.Population(
        30*8, p.extra_models.IFExpRateTwoComp(g_L=g_L), 
        label='population_3', in_partitions=[0, 0, 0, 0], out_partitions=3)

    dest = p.Population(
        1, p.extra_models.IFExpRateTwoComp(g_L=g_L), 
        label='dest', in_partitions=[4, 7, 3, 0], out_partitions=1)


    p.Projection(population1, dest, p.AllToAllConnector(),
                 receptor_type="soma_exc")
    p.Projection(population3, dest, p.AllToAllConnector(),
                 receptor_type="soma_inh")
    p.Projection(population2, dest, p.AllToAllConnector(),
                 receptor_type="dendrite_exc")

    dest.record(['v', 'gsyn_exc', 'gsyn_inh'])

    p.run(runtime)

    u = dest.get_data('v').segments[0].filter(name='v')[0]

    va = dest.get_data('gsyn_exc').segments[0].filter(name='gsyn_exc')[0]

    vb = dest.get_data('gsyn_inh').segments[0].filter(name='gsyn_inh')[0]

    p.end()

    return True


def success_desc():
    return "two level test PASSED"


def failure_desc():
    return "two level test FAILED"


if __name__ == "__main__":

    if test():
        print(success_desc())
    else:
        print(failure_desc())