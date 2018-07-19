from common_tools import data_summary, stats_setup, pickle_population
from basic_network import ConvMnistModel, MnistModel, NetworkModel, pool_init, evalModel
from deap import algorithms, base, creator, tools
import random
import numpy as np
import multiprocessing
from spinnman.exceptions import SpinnmanIOException, SpinnmanException
import pickle

from functools import partial


#GA and parallelisation variables

parallel_on = True
NUM_PROCESSES = 20 
IND_SIZE = (int(ConvMnistModel.filter_size**2)) + (ConvMnistModel.pop_1_size * ConvMnistModel.output_pop_size)
POP_SIZE = 100
NGEN = 100000 
toolbox = base.Toolbox()

#Setting up GA
creator.create("FitnessMin", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox.register("attribute", random.uniform, -10, 10)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalModel)
toolbox.register("mate", tools.cxTwoPoint)
toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
toolbox.register("select", tools.selBest)
CXPB = 0.5
MUTPB = 0.2
#the proportion of the population that is selected and mated
sel_factor = 10



#Statistics setup
logbook, mstats = stats_setup()
checkpoint_name = "logbooks/checkpoint.pkl"

def mAndM(population, toolbox, crossover_rate, mutation_rate, sel_factor):
    '''adapted from DEAP.algorithms.varAnd'''
    # create an offspring population the size of the population pre-selection
    offspring = [toolbox.clone(ind) for ind in population for i in range(0, sel_factor)]
    # shuffle offspring
    random.shuffle(offspring)
    
    # crossover
    for i in range(1, len(offspring), 2):
        if random.random() < crossover_rate:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
    # mutation
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    return offspring




def main(checkpoint = None):
    '''algorithm adapted from DEAP.algorithms.eaSimple'''

    global logbook
    try:
        with open(checkpoint, "r") as cp_file:
            cp = pickle.load(cp_file)
            pop = cp["population"]
            gen = cp["generation"]
            logbook = cp["logbook"]
            print("Checkpoint found... Generation %d" % gen)
    except IOError:
        print("No checkpoint found...")
        pop = toolbox.population(POP_SIZE)
        gen = 0
        print("Evaluating population")
        fitnesses = toolbox.map(toolbox.evaluate, pop)
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit
        pickle_population(pop, gen, logbook, checkpoint)
    
    for g in range(gen+1, NGEN):
        print ("Generation %d..." % g)
        print("Selecting %d from a population of %d..."% ( (len(pop)/sel_factor), len(pop)))
        offspring = toolbox.select(pop, (len(pop)/sel_factor))
        
        print("Applying crossover and mutation on the offspring...")
        offspring = mAndM(offspring, toolbox, CXPB, MUTPB, sel_factor)
        
        print("Evaluating the genes with an invalid fitness...")
        invalid_ind = [ind for ind in offspring if not ind.fitness.valid]
        fitnesses = toolbox.map(toolbox.evaluate, invalid_ind)
        
        if g==0:
            evals = POP_SIZE
        else:
            evals = 0
            
        for ind, fit in zip(invalid_ind, fitnesses):
            ind.fitness.values = fit
            evals += 1
            
        record = mstats.compile(pop)
        logbook.record(gen=g, evals=evals, **record)
        
        pickle_population(pop, g, logbook, checkpoint)
        
        print("Updating population...")
        pop[:] = offspring
    
    return;


if __name__ == "__main__":
   
    if parallel_on: 
	l=multiprocessing.Lock()
        pool = multiprocessing.Pool(NUM_PROCESSES, initializer=pool_init, initargs=(l,), maxtasksperchild=1)
        toolbox.register("map", pool.map)

    main(checkpoint_name)
    
    if not len(logbook)== 0:
        data_summary(logbook)
    
