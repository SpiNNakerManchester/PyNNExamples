import spynnaker8 as sim
import statistics
import math
from pyNN.utility.plotting import Figure, Panel
import matplotlib.pyplot as plt

def poisson_source_test():

    runtime = 100
    sim.setup(timestep=0.1)
    nNeurons = 128  # number of neurons in each population


    #rates = [400, 800, 1200]
    rng = sim.NumpyRNG(seed=1)

    weights = sim.RandomDistribution('uniform', [1.0, 3.0], rng=rng)

    cell_params_lif = {'cm': 0.25,
                       'i_offset': 0.0,
                       'tau_m': 20.0,
                       'tau_refrac': 2.0,
                       'tau_syn_E': 5.0,
                       'tau_syn_I': 5.0,
                       'v_reset': -70.0,
                       'v_rest': -65.0,
                       'v_thresh': -50.0
                       }

    weight_to_spike = 2.0

    rate = 400
    params = {"rate": rate, "poisson_weight": weights}

    pop = sim.Population(nNeurons, sim.IF_curr_exp(**cell_params_lif), label='pop')

    input = sim.Population(
        nNeurons, sim.PoissonSource, params, label='Poisson')

    pop.add_poisson_source(input)

    sim.run(runtime)

    sim.end()


if __name__ == "__main__":
    poisson_source_test()
