from common_tools import pool_init, evalModel, data_summary, stats_setup
from basic_network import MnistModel, NetworkModel
from deap import algorithms, base, creator, tools
import random
import numpy as np
import multiprocessing
from spinnman.exceptions import SpinnmanIOException, SpinnmanException
import pickle

from functools import partial


#GA and parallelisation variables

parallel_on = True
NUM_PROCESSES = 10
IND_SIZE = (MnistModel.input_pop_size * MnistModel.pop_1_size) + (MnistModel.pop_1_size * MnistModel.output_pop_size)
POP_SIZE = 20
NGEN = 320
toolbox = base.Toolbox()

#Setting up ES
creator.create("FitnessMin", base.Fitness, weights=(-1.0, -0.0001, 1.0))
creator.create("Gene", list, fitness=creator.FitnessMin)
creator.create("Strategy", np.array)

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

toolbox.register("mate", tools.cxESBlend, alpha=0.1)
toolbox.register("mutate", tools.mutESLogNormal, c=1.0, indpb=0.03)

toolbox.decorate("mate", checkStrategy(MIN_STRATEGY))
toolbox.decorate("mutate", checkStrategy(MIN_STRATEGY))



#Statistics setup
logbook, mstats = stats_setup()
checkpoint_name = "logbooks/ES.pkl"

def main(checkpoint = None):
    global logbook
    if checkpoint:
        from_pickle = True
        with open(checkpoint, "r") as cp_file:
            cp = pickle.load(cp_file)
            pop = cp["population"]
            gen = cp["generation"]
            logbook = cp["logbook"]
            print("Checkpoint found... Generation %d" % gen)
    else:
        print("No checkpoint given...")
        pop = toolbox.population(POP_SIZE)
        gen = 0
        from_pickle = False
    
    if not from_pickle:
        print("Evaluating population")
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        pickle_population(pop, gen, logbook, checkpoint_name)
    
    MU, LAMBDA = 10, 100
    pop = toolbox.population(n=MU)
    hof = tools.HallOfFame(1)
    stats = tools.Statistics(lambda ind: ind.fitness.values)
    stats.register("avg", numpy.mean)
    stats.register("std", numpy.std)
    stats.register("min", numpy.min)
    stats.register("max", numpy.max)
    
    pop, logbook = algorithms.eaMuCommaLambda(pop, toolbox, mu=MU, lambda_=LAMBDA, 
        cxpb=0.6, mutpb=0.3, ngen=500, stats=stats, halloffame=hof)
    
    return pop, logbook, hof


if __name__ == "__main__":
    
    if parallel_on:
        pool = multiprocessing.Pool(NUM_PROCESSES, initializer=pool_init, maxtasksperchild=1)
        toolbox.register("map", pool.map)

    main(checkpoint_name)
    
    if not len(logbook)== 0:
        data_summary(logbook)
    