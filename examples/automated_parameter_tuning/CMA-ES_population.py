from common_tools import flatten_fitnesses, data_summary, stats_setup, pickle_population, pickle_strategy, split_population, split_fit
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
NUM_PROCESSES = 100 
IND_SIZE = (int(ConvMnistModel.filter_size**2)) + (ConvMnistModel.pop_1_size * ConvMnistModel.output_pop_size)
NGEN = 100
SUBPOP_SIZE = 171
#240 = 5 networks per chip * 48 chips per board

toolbox = base.Toolbox()

#Setting up EA
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox.register("attribute", random.uniform, -1.0, 1.0)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)

#Statistics setup
logbook, mstats = stats_setup()
checkpoint_name = "logbooks/CMA-ES_checkpoint.pkl"

def eaGenerateUpdate(checkpoint, strategy, logbook, toolbox, gen, ngen, halloffame=None, stats=None,
                     verbose=__debug__):
    '''This function is a derivative of the eaGenerateUpdate in the DEAP library
    and has been adapted to allow the population to be split into subpopulations
    to be evaluated simulataneously on SpiNNaker.
    '''
        
    for g in range(gen+1, ngen+1):
        # Generate a new population
        population = toolbox.generate()
        
        sub_pops = split_population(population, SUBPOP_SIZE)
        toolbox.register("evaluate", evalPopulation, g)
        # Evaluate the individuals
        fitnesses_and_times_eval = toolbox.map(toolbox.evaluate, sub_pops)
        fitnesses, times = split_fit(fitnesses_and_times_eval)
        gc.collect()
        print("Adding fitness values to individuals...") 
        for ind, fit in zip(population, fitnesses):
            ind.fitness.values = fit,
	print("Making hof...")
        if halloffame is not None:
            halloffame.update(population)
	print("Updating strategy")
        # Update the strategy with the evaluated individuals
        toolbox.update(population)
        print("Recording stats")
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=g, nevals=len(population), **record)
        if verbose:
            print logbook.stream
        print("Pickling strategy")
        pickle_strategy(strategy, g, logbook, checkpoint)
        gc.collect()
    return logbook

def main(checkpoint = None):
    '''algorithm adapted from DEAP.algorithms.eaSimple'''
    global logbook
    np.random.seed(42)
    N = IND_SIZE
    hof = tools.HallOfFame(1)
        
    try:
        with open(checkpoint, "r") as cp_file:
            cp = pickle.load(cp_file)
            gen = cp["generation"]
            logbook = cp["logbook"]
            strategy = cp["strategy"]
            print("Checkpoint found... Generation %d" % gen)
    except IOError:
        print("No checkpoint found...")
        gen = 0
        logbook = tools.Logbook()
        strategy = cma.Strategy(centroid=[0.0]*N, sigma=0.5)
    
    toolbox.register("generate", strategy.generate, creator.Individual)
    toolbox.register("update", strategy.update)

    logbook = eaGenerateUpdate(checkpoint, strategy, logbook, toolbox, gen=gen, ngen=NGEN, stats=mstats, halloffame=hof)
    
    return logbook;


if __name__ == "__main__":
   
    if parallel_on: 
        l=multiprocessing.Lock()
        pool = multiprocessing.Pool(NUM_PROCESSES, initializer=pool_init, initargs=(l,), maxtasksperchild=1)
        toolbox.register("map", pool.map)

    logbook = main(checkpoint_name)
    
    if not len(logbook)== 0:
        data_summary(logbook)
    
