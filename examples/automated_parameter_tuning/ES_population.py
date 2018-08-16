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
import array
from itertools import repeat
#To supress warnings about having packages compiled against earlier numpy version
import warnings
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from functools import partial

'''This code builds upon example code given in the DEAP documentation'''

#GA and parallelisation variables

parallel_on = True
NUM_PROCESSES = 2 
IND_SIZE = (int(ConvMnistModel.filter_size**2)) + (ConvMnistModel.pop_1_size * ConvMnistModel.output_pop_size)
NGEN = 100
SUBPOP_SIZE = 171
MIN_VALUE = -10
MAX_VALUE = 10
MIN_STRATEGY = 0.01
MAX_STRATEGY = 3

#240 = 5 networks per chip * 48 chips per board

toolbox = base.Toolbox()

#Setting up EA
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", array.array, typecode="d", fitness=creator.Fitness, strategy=None)
creator.create("Strategy", array.array, typecode="d")


#Statistics setup
logbook, mstats = stats_setup()
checkpoint_name = "logbooks/ES_checkpoint.pkl"


# Individual generator
def generateES(icls, scls, size, imin, imax, smin, smax):
    ind = icls(random.uniform(imin, imax) for _ in range(size))
    ind.strategy = scls(random.uniform(smin, smax) for _ in range(size))
    return ind

def checkStrategy(minstrategy):
    def decorator(func):
        def wrappper(*args, **kargs):
            children = func(*args, **kargs)
            for child in children:
                for i, s in enumerate(child.strategy):
                    if s < minstrategy:
                        child.strategy[i] = minstrategy
            return children
        return wrappper
    return decorator

def eaMuCommaLambda(population, toolbox, mu, lambda_, cxpb, mutpb, ngen, checkpoint,
                    stats=None, halloffame=None, verbose=__debug__):
    '''Adapted from eaMuCommaLambda in DEAP library'''
    global logbook
    
    assert lambda_ >= mu, "lambda must be greater or equal to mu."

    # Evaluate the individuals with an invalid fitness
    invalid_ind = [ind for ind in population if not ind.fitness.valid]
    sub_pops = split_population(population, SUBPOP_SIZE)
    toolbox.register("evaluate", evalPopulation, 1)
    fitnesses_and_times_eval = toolbox.map(toolbox.evaluate, sub_pops)
    fitnesses, times = split_fit(fitnesses_and_times_eval)
    
    for ind, fit in zip(invalid_ind, fitnesses):
        ind.fitness.values = fit

    if halloffame is not None:
        halloffame.update(population)

    record = stats.compile(population) if stats is not None else {}
    logbook.record(gen=1, nevals=len(invalid_ind), **record)
    if verbose:
        print logbook.stream
        
    pickle_population(population, 1, logbook, checkpoint)

    # Begin the generational process
    for gen in range(gen+1, ngen + 1):
        # Vary the population
        offspring = varOr(population, toolbox, lambda_, cxpb, mutpb)

        # Evaluate the individuals with an invalid fitness
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        toolbox.register("evaluate", evalPopulation, 0)
        fitnesses_and_times_eval = toolbox.map(toolbox.evaluate, sub_pops)
        fitnesses, times = split_fit(fitnesses_and_times_eval)
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit

        # Update the hall of fame with the generated individuals
        if halloffame is not None:
            halloffame.update(offspring)

        # Select the next generation population
        population[:] = toolbox.select(offspring, mu)

        # Update the statistics with the new population
        record = stats.compile(population) if stats is not None else {}
        logbook.record(gen=gen, nevals=len(invalid_ind), **record)
        if verbose:
            print logbook.stream
    return population, logbook

toolbox = base.Toolbox()
toolbox.register("individual", generateES, creator.Individual, creator.Strategy,
    IND_SIZE, MIN_VALUE, MAX_VALUE, MIN_STRATEGY, MAX_STRATEGY)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)
toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)
toolbox.register("select", tools.selTournament, tournsize=3)

toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))

def main(checkpoint = None):
    random.seed()
    MU, LAMBDA = 10, 100
    global logbook
    try:
        with open(checkpoint, "r") as cp_file:
            cp = pickle.load(cp_file)
            gen = cp["generation"]
            pop = cp["population"]
            logbook = cp["logbook"]
            print("Checkpoint found... Generation %d" % gen)
    except IOError:
        print("No checkpoint found...")
        gen = 1
        pop = toolbox.population(n=MU)

        pop, logbook = eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
        cxpb=0.6, mutpb=0.3, ngen=500, checkpoint=checkpoint, stats=mstats)
    
    return pop, logbook, hof

if __name__ == "__main__":
   
    if parallel_on: 
        l=multiprocessing.Lock()
        pool = multiprocessing.Pool(NUM_PROCESSES, initializer=pool_init, initargs=(l,), maxtasksperchild=1)
        toolbox.register("map", pool.map)

    pop, logbook, hof = main(checkpoint_name)
    
    if not len(logbook)== 0:
        data_summary(logbook)
    
