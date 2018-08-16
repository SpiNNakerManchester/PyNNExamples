from common_tools import flatten_fitnesses, data_summary, stats_setup, pickle_population, split_population, average_times, write_csv_data_file, split_fit
from basic_network import ConvMnistModel, MnistModel, NetworkModel, pool_init, evalModel, evalPopulation, timer
from deap import algorithms, base, creator, tools
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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument("checkpoint")
args = parser.parse_args()
checkpoint_name = args.checkpoint



#GA and parallelisation variables

parallel_on = True
NUM_PROCESSES = 100 
IND_SIZE = (int(ConvMnistModel.filter_size**2)) + (ConvMnistModel.pop_1_size * ConvMnistModel.output_pop_size)
POP_SIZE = 2400
#NGEN = 1000000
SUBPOP_SIZE = 171 
#240 = 5 networks per chip * 48 chips per board

toolbox = base.Toolbox()

#Setting up GA
creator.create("Fitness", base.Fitness, weights=(1.0,))
creator.create("Individual", list, fitness=creator.Fitness)

#For continuous networks
#toolbox.register("attribute", random.uniform, -10, 10)
#Reduced parameter space by restricting weights
toolbox.register("attribute", random.randint, -1, 1)
toolbox.register("individual", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.individual)

toolbox.register("evaluate", evalModel)
toolbox.register("mate", tools.cxTwoPoint)
#for continuous networks
#toolbox.register("mutate", tools.mutGaussian, mu=0.0, sigma=0.2, indpb=0.2)
toolbox.register("mutate", tools.mutUniformInt, low=-1,up=1, indpb=0.001)
toolbox.register("select", tools.selBest)
CXPB = 0.5
MUTPB = 1
#the proportion of the population that is selected and mated
sel_factor = 10

#Statistics setup
logbook, mstats = stats_setup()

def mAndM(population, toolbox, crossover_rate, mutation_rate, sel_factor):
    '''adapted from DEAP.algorithms.varAnd'''
    # create an offspring population the size of the population pre-selection
    offspring_elite = [toolbox.clone(ind) for ind in population]
    offspring = [toolbox.clone(ind) for ind in population for i in range(0, sel_factor-1)]
    # shuffle offspring
    random.shuffle(offspring)
    # crossover
    print("crossing over")
    for i in range(0, len(offspring), 2):
        if random.random() < crossover_rate:
            offspring[i - 1], offspring[i] = toolbox.mate(offspring[i - 1], offspring[i])
            del offspring[i - 1].fitness.values, offspring[i].fitness.values
    # mutation
    print("mutating")
    for i in range(len(offspring)):
        if random.random() < mutation_rate:
            offspring[i], = toolbox.mutate(offspring[i])
            del offspring[i].fitness.values

    offspring.extend(offspring_elite)
    return offspring;


def main(generations, checkpoint = None):
    '''algorithm adapted from DEAP.algorithms.eaSimple'''
    NGEN = generations
    global logbook
    try:
        with open(checkpoint, "r") as cp_file:
            cp = pickle.load(cp_file)
            pop = cp["population"]
            gen = cp["generation"]
            logbook = cp["logbook"]
            print("Checkpoint found... Generation %d" % gen)
            print("Population size %s" % len(pop))
    except IOError:
        print("No checkpoint found...")
        print("Generating population of size %s" % POP_SIZE)
        pop = toolbox.population(POP_SIZE)
        gen = 0
        print("Evaluating Generation 0")
        toolbox.register("evaluatepop", evalPopulation, gen)
        pop_split = split_population(pop, SUBPOP_SIZE)
        fitnesses_and_times_eval = toolbox.map(toolbox.evaluatepop, pop_split)        
        fitnesses, times = split_fit(fitnesses_and_times_eval)
        gc.collect()
        for ind, fit in zip(pop, fitnesses):
            ind.fitness.values = fit,
        pickle_population(pop, gen, logbook, checkpoint)
        gc.collect()
        
    for g in range(gen+1, NGEN+1):
        global SUBPOP_SIZE
	print("ok")
        print ("Generation %d..." % g)
        t_start_gen = timer()
        print("Selecting %d from a population of %d..."% ( (len(pop)/sel_factor), len(pop)))
        offspring = toolbox.select(pop, (len(pop)/sel_factor))
        t_end_select = timer()
        print("Applying crossover and mutation on the offspring...")
        offspring = mAndM(offspring, toolbox, CXPB, MUTPB, sel_factor)
        t_end_variation = timer()
        print("Evaluating the genes with an invalid fitness...")
        toolbox.register("evaluatepop", evalPopulation, g)
        offspring_split = split_population(offspring, SUBPOP_SIZE)
        t_end_pop_preprocess = timer()
        fitnesses_and_times_eval = toolbox.map(toolbox.evaluatepop, offspring_split)
        fitnesses, times = split_fit(fitnesses_and_times_eval)
        t_end_evaluatepop = timer()
        gc.collect()
                    
        for ind, fit in zip(offspring, fitnesses):
            ind.fitness.values = fit,
            
        print("Updating population...")
        pop[:] = offspring
        t_end_pop_postprocess = timer()
        print("Recording stats")
        record = mstats.compile(pop)
        logbook.record(gen=g, evals=len(offspring), **record)
        t_end_stats = timer()
        print("Pickling population...")
        pickle_population(pop, g, logbook, checkpoint)
        gc.collect()
        t_end_gen = timer()
        avg_times_eval = average_times(times, SUBPOP_SIZE)
        times_gen = (t_start_gen, t_end_select, t_end_variation, t_end_pop_preprocess, t_end_evaluatepop, t_end_pop_postprocess, t_end_stats, t_end_gen)
        #t_start_gen, t_end_select, t_end_variation, t_end_pop_preprocess, t_end_evaluatepop, t_end_pop_postprocess, t_end_stats, t_end_gen, number_evals, t_min, t_setup, t_run, t_gather, t_cost, avg_retry  
        total_data = (SUBPOP_SIZE, POP_SIZE, NUM_PROCESSES) + times_gen + avg_times_eval
        write_csv_data_file(total_data, "timing_data.csv")
        print("data written to file")
	#SUBPOP_SIZE = SUBPOP_SIZE + 2020202020202020202020202020202020202020 
	#print("SUBPOP_SIZE increased to %s" % SUBPOP_SIZE)
    return;


if __name__ == "__main__":
   
    if parallel_on: 
        l=multiprocessing.Lock()
        pool = multiprocessing.Pool(NUM_PROCESSES, initializer=pool_init, initargs=(l,), maxtasksperchild=1)
        toolbox.register("map", pool.map)

	main(100, checkpoint_name)
    
    if not len(logbook)== 0:
        data_summary(logbook)
    
