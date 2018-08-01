from deap import algorithms, base, creator, tools
from basic_network import ConvMnistModel, MnistModel, NetworkModel
import pickle
import random
import time
import multiprocessing
import numpy as np
import sys
import gc
import matplotlib.pyplot as plt

def split_population(pop, subpop_size, gen):
    '''splits a population into a number of subpopulations of subpop_size'''
    #ceiling division
    number_subpops = -(-len(pop)//subpop_size)
    subpops = []
    for i in range (0, len(pop)+1, subpop_size):
        subpops.append((pop[i:i+1], gen))
    return subpops;

def pickle_population(pop, gen, log, checkpoint_name):
    print("Pickling population from generation %d..." % gen)
    cp = dict(population=pop, generation=gen, logbook=log)

    with open(checkpoint_name, "wb") as cp_file:
        pickle.dump(cp, cp_file)
    return;

def data_summary(logbook):
    print(logbook)
    gen = logbook.select("gen")
    
    max_accuracy = logbook.chapters["accuracy"].select("max")
    min_accuracy = logbook.chapters["accuracy"].select("min")
    avg_accuracy = logbook.chapters["accuracy"].select("avg")
    avg_ce = logbook.chapters["cross_entropy"].select("avg")
    avg_norm = logbook.chapters["norm"].select("avg")
    
    #weights = (-1.0, -1.0, 2.0)
    
    #fitness = (weights[0]*np.asfarray(avg_ce)) + (weights[1]*np.asfarray(avg_norm)) + (weights[2]*np.asfarray(avg_accuracy))
    
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, avg_accuracy, "g-", label = "mean accuracy")
    line2 = ax1.plot(gen, max_accuracy, "r-", label= "maximum accuracy")
    line3 = ax1.plot(gen, min_accuracy, "b-", label="minimum accuracy")
    ax1.set_xlabel("Generation")
    ax1.set_ylabel("Accuracy")
    plt.legend()
    #ax1.xticks(np.arange(0, max(gen)+1, 1.0))
    for tl in ax1.get_yticklabels():
        tl.set_color("b")
    plt.show()
    return

def stats_setup():
    logbook = tools.Logbook()
    #stats_ce = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    #stats_norm = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    stats_accuracy = tools.Statistics(key=lambda ind: ind.fitness.values)
    mstats = tools.MultiStatistics(accuracy=stats_accuracy)
    mstats.register("avg", np.mean, axis=0)
    mstats.register("std", np.std, axis=0)
    mstats.register("min", np.min, axis=0)
    mstats.register("max", np.max, axis=0)
    return logbook, mstats
'''
multiobjective
def stats_setup():
    logbook = tools.Logbook()
    stats_ce = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    stats_norm = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    stats_accuracy = tools.Statistics(key=lambda ind: ind.fitness.values[2])
    mstats = tools.MultiStatistics(cross_entropy=stats_ce, norm=stats_norm, accuracy=stats_accuracy)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
    return logbook, mstats'''
