'''A tool to allow pickled data to be viewed'''
import argparse
from deap import algorithms, base, creator, tools
from basic_network import  ConvMnistModel, MnistModel, NetworkModel,pool_init, evalModel
import random
from common_tools import data_summary, stats_setup, pickle_population

import pickle
from common_tools import data_summary

IND_SIZE = (int(ConvMnistModel.filter_size**2)) + (ConvMnistModel.pop_1_size * ConvMnistModel.output_pop_size)
toolbox = base.Toolbox()

#Setting up GA
creator.create("FitnessMin", base.Fitness, weights=(1.0))
creator.create("Individual", list, fitness=creator.FitnessMin)

toolbox.register("attribute", random.uniform, -10, 10)
toolbox.register("gene", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.gene)


#Statistics setup
logbook, mstats = stats_setup()



checkpoint = "logbooks/checkpoint.pkl"

try:
    with open(checkpoint, "r") as cp_file:
        cp = pickle.load(cp_file)
        pop = cp["population"]
        gen = cp["generation"]
        logbook = cp["logbook"]
        print("Checkpoint found... Generation %d" % gen)        
        
        testModel = MnistModel(pop[0])
        testModel.visualise_filter()
        #testModel.visualise_input_weights()
        #testModel.visualise_output_weights()
        
except IOError:
    print("No checkpoint found...")

data_summary(logbook)



