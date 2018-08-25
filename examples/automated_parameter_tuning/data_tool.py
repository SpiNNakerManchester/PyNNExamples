'''A tool to allow pickled data to be viewed'''
import matplotlib.pyplot as plt
from sklearn.metrics import euclidean_distances
from sklearn.manifold import MDS
import pandas as pd
import argparse
import multiprocessing
from deap import algorithms, base, creator, tools
from basic_network import  ConvMnistModel, MnistModel, NetworkModel,pool_init, evalModel
import random
from common_tools import data_summary, stats_setup, pickle_population, write_csv_logbook_file
import cPickle as pickle
import numpy as np
from sphinx.domains import std
from pyNN.space import distance

IND_SIZE = (int(ConvMnistModel.filter_size**2)) + (ConvMnistModel.pop_1_size * ConvMnistModel.output_pop_size)
toolbox = base.Toolbox()

#Setting up GA
creator.create("Fitness", base.Fitness, weights=(1.0))
creator.create("Individual", list, fitness=creator.Fitness)

toolbox.register("attribute", random.randint, -1, 1)
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
        im = ax.imshow(img, cmap="gray_r")
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

def average_average_filter():
    avg_filters = []

    
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
        
        avg_filters.append(average_filter(pop, 5))
        
    avg_filters = np.array(avg_filters)
    avg_avg_filter = np.average(avg_filters, axis=0)
    ConvMnistModel.visualise_filter(np.reshape(avg_avg_filter, (5,5)), 5)
    
    
    
    
    return avg_filters


def save_logbook_to_csv(logbook, checkpoint):
    gen = np.array(logbook.chapters["accuracy"].select("avg"))
    avg = np.array(logbook.chapters["accuracy"].select("avg"))
    max = np.array(logbook.chapters["accuracy"].select("max"))
    min = np.array(logbook.chapters["accuracy"].select("min"))
    std = np.array(logbook.chapters["accuracy"].select("std"))
    
    data = np.hstack((avg, max, min, std))

    label_list = ["avg", "max", "min", "std"]
    df_logbook = pd.DataFrame(data)
    
    df_logbook.columns = label_list
    df_logbook.transpose()

    filename = checkpoint + ".csv"      
    
    df_logbook.to_csv(filename, index=False)
    return;
    
def save_fitnesses_to_csv(pop, checkpoint):
    fitnesses = []
    for ind in pop:
        fitnesses.append(ind.fitness.values)
    
    print(fitnesses)
    
    fitnesses = pd.DataFrame(fitnesses)
    filename = checkpoint + "_fitnesses.csv"  
    fitnesses.to_csv(checkpoint, index=False)
    return

def multiple_filter_visualisation(pop):
    filters = []
    for ind in pop:
        filter_colours = []
        for i in range(5):
            trit = ind[(5*i):5*(i+1)]
            sum = 0
            for j in range(5):
                sum+=trit[j]*(3^j)
            filter_colours.append(sum)
        filters.append(filter_colours)
    filters = np.array(filters)
    filters = np.reshape(filters,(1200,100))
    plt.imshow(filters)
    plt.show()
    return;   

def population_distances(pop):
    pop = np.array(pop)
    pop = np.int8(pop)
    return euclidean_distances(pop);

#average_average_filter()

'''
#Statistics setup
logbook, mstats = stats_setup()

#visualise_multiple_populations()
    
##
'''
checkpoint = "logbooks/pop_24000_1g_init_pattern_6000b_6000-b.pkl"
distance_file = "logbooks/distances.pkl"

try:
    with open(checkpoint, "r") as cp_file:
        print("loading pickled file")
        cp = pickle.load(cp_file)
        print("loading pickled population")
        pop = cp["population"]
        #gen = cp["generation"]
        #logbook = cp["logbook"]
        print("visualising")
        try:
            with open(distance_file, "r") as cp_file:
                print("distances found")
                distances = pickle.load(cp_file)
                
        except IOError:
                print("no distances, generating them...")
                distances = population_distances(pop)
                pickle.dump(distances, open(distance_file, "wb" ))
        print("doing the MDS")
        mds = MDS(2, False, max_iter=2, n_init=1)
        Y = mds.fit_transform(X)
        pickle.dump(Y, open("logbooks/plot_output.pkl", "wb"))
        
        plt.scatter(Y[:, 0], Y[:, 1], c=color, cmap=plt.cm.Spectral)
        plt.show()
        
        
        #multiple_filter_visualisation(pop)
        #print("logbook loaded")
        #print(logbook)
        #save_logbook_to_csv(logbook, checkpoint)
        #save_fitnesses_to_csv(pop, checkpoint)
        
        #ConvMnistModel.visualise_filter(average_filter(pop, 5), 5)
        #best = characterise_best(pop)
        #print(best)
        #ConvMnistModel.visualise_filter(pop[best][:25], 5)
        #testModel = ConvMnistModel(pop[9], True)
        #testModel.visualise_filter()
        #testModel.visualise_input_weights()
        #testModel.visualise_output_weights()
        
except IOError:
    print("No checkpoint found...")

#data_summary(logbook)
'''
'''

