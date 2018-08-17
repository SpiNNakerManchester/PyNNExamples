'''A tool to allow pickled data to be viewed'''
import matplotlib.pyplot as plt

import argparse
import multiprocessing
from deap import algorithms, base, creator, tools
from basic_network import  ConvMnistModel, MnistModel, NetworkModel,pool_init, evalModel
import random
from common_tools import data_summary, stats_setup, pickle_population
import pickle
import numpy as np

IND_SIZE = (int(ConvMnistModel.filter_size**2)) + (ConvMnistModel.pop_1_size * ConvMnistModel.output_pop_size)
toolbox = base.Toolbox()

#Setting up GA
creator.create("Fitness", base.Fitness, weights=(1.0))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox.register("attribute", random.uniform, -10, 10)
toolbox.register("gene", tools.initRepeat, creator.Individual, toolbox.attribute, n=IND_SIZE)
toolbox.register("population", tools.initRepeat, list, toolbox.gene)


def average_filter(pop, filter_size):
    filters = [[] for i in pop]
    for i in range(len(pop)):
        filters[i]= pop[i][:(filter_size**2)]
    
    filters = np.asarray([np.array(x) for x in filters])
   
    
    avg_filter = np.average(filters, axis=0)

    #print(avg_filter)
    #ConvMnistModel.visualise_filter(avg_filter,5)
    return avg_filter

def characterise_best(pop):
    best = 0
    for i in range(len(pop)):
        if pop[i].fitness > pop[best].fitness:
            best=i
    return best;

def visualise_multiple(poplist, accuracy_list):
    
    fig, axes = plt.subplots(nrows=2, ncols=5)
    
    for i, (ax, img) in enumerate(zip(axes.flat, poplist)):
        im = ax.imshow(img, cmap="gray")
        ax.set_title("Population " + str(i) + ", accuracy: " + str(100*round(accuracy_list[i],3)) +"%")


    fig.subplots_adjust(right=0.8)
    cbar_ax = fig.add_axes([0.85, 0.15, 0.05, 0.7])
    fig.colorbar(im, cax=cbar_ax)
    
    plt.show()  

    return;

def visualise_multiple_populations():
    avg_filters = []
    best_filters = []
    best_accuracies = []
    average_accuracies = []
    
    for i in range(1,11):
        checkpoint="logbooks/logbooks/small_tests/checkpoint_population_" +str(i) +".pkl"
        try:
            with open(checkpoint, "r") as cp_file:
                print("loading pickled file")
                cp = pickle.load(cp_file)
                print("loading pickled population")
                pop = cp["population"]
                gen = cp["generation"]
                logbook = cp["logbook"]
                print("Checkpoint found... Generation %d" % gen)        
                
        except IOError:
            print("No checkpoint found...")  
        
        avg_filters.append(np.reshape(average_filter(pop, 5), (5,5)))
        best = characterise_best(pop)
        best_filters.append(np.reshape(pop[best][:25], (5,5)))
        best_accuracies.append(logbook.chapters["accuracy"].select("max")[-1][0])
        average_accuracies.append(logbook.chapters["accuracy"].select("avg")[-1][0])
    
        print checkpoint
    
    visualise_multiple(best_filters, best_accuracies)
    visualise_multiple(avg_filters, average_accuracies)
    return



#Statistics setup
logbook, mstats = stats_setup()

    
##

checkpoint = "logbooks/logbooks/small_tests/.pkl"

try:
    with open(checkpoint, "r") as cp_file:
        print("loading pickled file")
        cp = pickle.load(cp_file)
        print("loading pickled population")
        pop = cp["population"]
        gen = cp["generation"]
        logbook = cp["logbook"]
        print("Checkpoint found... Generation %d" % gen)        
        average_filter(pop, 5)
        best = characterise_best(pop)
        print(best)
        ConvMnistModel.visualise_filter(pop[best][:25], 5)
        #testModel = ConvMnistModel(pop[9], True)
        #testModel.visualise_filter()
        #testModel.visualise_input_weights()
        #testModel.visualise_output_weights()
        
except IOError:
    print("No checkpoint found...")

data_summary(logbook)

'''


