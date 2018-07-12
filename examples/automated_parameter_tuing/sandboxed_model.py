import pyNN.utility.plotting as plot
import matplotlib.pyplot as plt
from random import randint
import numpy as np
from pyNN.random import RandomDistribution
from spinnman.exceptions import SpinnmanIOException


num_retries = 0
max_retries = 5

def run_sim():
    global num_retries
    try:
        import pyNN.spiNNaker as p
        p.setup(timestep=1.0)
        p.set_number_of_neurons_per_core(p.IF_curr_exp,100)
       
        pop_1 = p.Population(100, p.IF_curr_exp(), label="pop_1")
        input_pop = p.Population(1, p.SpikeSourcePoisson(), label="input")
        input_proj = p.Projection(input_pop, pop_1, p.FromListConnector([(0, 0)]),
                                    synapse_type=p.StaticSynapse(weight=5, delay=1))
       
        size = 99
        connection_list =[(size, 0)]
        for i in range(0, size+1):
            new_connection = (i,i+1)
            connection_list.append(new_connection)
           
       
        randomise = RandomDistribution('normal_clipped_to_boundary', mu=i, sigma=size/5, low=0, high=size)
        syn_fire_proj = p.Projection(pop_1, pop_1, p.FromListConnector(connection_list), synapse_type=p.StaticSynapse(weight=5, delay=1))
        pop_1.record(["spikes"])
        simtime = 2000
        p.run(simtime)
        neo = pop_1.get_data(variables=["spikes"])
        p.end()
        spikes = neo.segments[0].spiketrains
        print spikes
    except SpinnmanIOException as e:
        print(e)
        if num_retries < max_retries:
            num_retries += 1
            print("Retry %d..." % num_retries)
            run_sim()
    return;

run_sim()