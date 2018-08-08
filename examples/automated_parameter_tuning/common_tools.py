import os
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
import warnings
import csv
warnings.filterwarnings("ignore", message="numpy.dtype size changed")
warnings.filterwarnings("ignore", message="numpy.ufunc size changed")
from mnist import MNIST
#sys.path.append('/localhome/mbaxsej2/optimisation_env/NE15')
home = os.path.expanduser("~")
#home = os.environ['VIRTUAL_ENV']
NE15_path = home + '/git/NE15'
sys.path.append(NE15_path)


def set_up_training_data():
    #picking test images from data 
    number_digits = 10
    
    mndata = MNIST(home)
    mndata.gz = True
    images, labels = mndata.load_training()
    
    test_data = [[] for i in range(number_digits)]
    
    for label, image in zip(labels, images):
            test_data[label].append(image)
            
    print(len(test_data))
    print([len(list) for list in test_data])
            
    data_filename = 'training_data/processed_training_data.pkl'
    outfile = open(data_filename,'wb')
    pickle.dump(test_data, outfile)
    outfile.close()
    
    return;

def flatten_fitnesses(fitnesses):
    fitnesses_final = []
    for fitnesslist in fitnesses:
        fitnesses_final.extend(fitnesslist)
    
    return fitnesses_final;

def split_population(pop, subpop_size, gen):
    '''splits a population into a number of subpopulations of subpop_size'''
    #ceiling division
    number_subpops = -(-len(pop)//subpop_size)
    subpops = []
    for i in range (0, len(pop)+1, subpop_size):
        subpops.append((pop[i:i+subpop_size], gen))
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


test_times = [(1,1,2,3,5,240,4), (2,2,3,6,10,240,1), (4,4,6,10,22,100,0)]


def average_times(times, subpop):
    #(t_start_eval, t_end_setup, t_end_run, t_end_gather, t_end_eval, len(pop), num_retries)
    times = np.array(times)
    times_original = times.copy()
    number_evals = times.shape[1]
    t_min = np.amin(times[:,0])
    #get non-remainders
    times = times[times[:,-2]== subpop]
    other_stats = times[-2:,:]
    times = times[:, :-2]
    times = np.diff(times)
    avg_times = np.average(times, axis=0).tolist()
    avg_retry = np.average(times_original[:,-1])
    avg_times = (number_evals, t_min,) + tuple(avg_times) + (avg_retry,)
    #(t_min, t_setup, t_run, t_gather, t_cost, avg_retry)    
    return avg_times;

#print(average_times(test_times, 240))
#set_up_training_data()

def split_fit(fit_times):
    fitnesses = []
    times = []
    
    for i in fit_times:
        fitnesses.extend(i[0])
        times.append(i[1])

    return fitnesses, times

def write_csv_data_file(data, filename):
    data = list(data)
    with open(filename, 'a') as file:
        writer = csv.writer(file)
        writer.writerow(data)

    file.close()
    return;

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
