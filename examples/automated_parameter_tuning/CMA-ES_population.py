from common_tools import flatten_fitnesses, data_summary, stats_setup, pickle_population, split_population
from basic_network import ConvMnistModel, MnistModel, NetworkModel, pool_init, evalModel, evalPopulation
from deap import algorithms, base, creator, tools, cma
import random
import numpy as np
import multiprocessing
from spinnman.exceptions import SpinnmanIOException, SpinnmanException
import pickle
import pprint
import gc
from itertools import repeat
#To supress warnings about having packages compiled against earlier numpy version
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from functools import partial

'''This code builds upon example code given in the DEAP documentation'''

#GA and parallelisation variables

parallel_on = True
NUM_PROCESSES = 10
IND_SIZE = (int(ConvMnistModel.filter_size**2)) + (ConvMnistModel.pop_1_size * ConvMnistModel.output_pop_size)
NGEN = 250
SUBPOP_SIZE = 240
#240 = 5 networks per chip * 48 chips per board

toolbox = base.Toolbox()

#Setting up EA
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox.register("attribute", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
toolbox.register("evaluate", evalPopulation)

#Statistics setup
logbook, mstats = stats_setup()
checkpoint_name = "logbooks/CMA-ES_checkpoint.pkl"

def eaGenerateUpdate(toolbox, ngen, halloffame=None, stats=None,
                     verbose=__debug__):
    '''This function is a derivative of the eaGenerateUpdate in the DEAP library
    and has been adapted to allow the population to be split into subpopulations
    to be evaluated simulataneously on SpiNNaker.
    '''
    logbook = tools.Logbook()
    logbook.header = ['gen', 'nevals'] + (stats.fields if stats else [])

    for gen in range(ngen):
        # Generate a new population
        population = toolbox.generate()
        
        sub_pops = split_population(population, SUBPOP_SIZE)
        toolbox.register("evaluatepop", evalPopulation, gen)
        # Evaluate the individuals
        fitnesses_and_times_eval = toolbox.map(toolbox.evaluate, subpops)
        fitnesses, times = split_fit(fitnesses_and_times_eval)
        gc.collect()
        
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit

        if halloffame is not None:
            halloffame.update(population)

        # Update the strategy with the evaluated individuals
        toolbox.update(population)

        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(population), **record)
        if verbose:
            print logbook.stream

return population, logb



def main(checkpoint = None):
    '''algorithm adapted from DEAP.algorithms.eaSimple'''
     
    np.random.seed(42)
    N = IND_SIZE
    strategy = cma.Strategy(centroid=[5.0]*N, sigma=5.0, lambda_=20*N)
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", np.mean)
    stats.register("std", np.std)
    stats.register("min", np.min)
    stats.register("max", np.max)
    algorithms.eaGenerateUpdate(toolbox, ngen=NGEN, stats=stats, halloffame=hof)

    gc.collect()
    return;


if __name__ == "__main__":
   
    if parallel_on: 
        l=multiprocessing.Lock()
        pool = multiprocessing.Pool(NUM_PROCESSES, initializer=pool_init, initargs=(l,), maxtasksperchild=1)
        toolbox.register("map", pool.map)

    main(checkpoint_name)
    
    if not len(logbook)== 0:
        data_summary(logbook)
    
