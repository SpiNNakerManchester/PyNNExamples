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

def pool_init():  
    gc.collect()
    #get lock would be better
    #time.sleep(random.randint(0,20))
    return;

def evalModel(gene):
    '''evaluates the model'''
    gc.collect()
    current = multiprocessing.current_process()
    print ("Process " + current.name + " started.")
    f_name = "errorlog/" + current.name +"_stdout.txt"
    g_name = "errorlog/" + current.name + "_stderror.txt"
    f = open(f_name, 'w')
    g = open(g_name, 'w')
    old_stdout = sys.stdout
    old_stderr = sys.stderr
    sys.stdout = f
    sys.stderr = g
    try:
        model = ConvMnistModel(gene)
        model.test_model()
        sys.stdout = old_stdout
        sys.stderr = old_stderr            
        print ("Process " + current.name + " finished sucessfully: %s" % (model.cost,)) 
        return model.cost;
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print ("Process " + current.name + " stopped unexpectedly.") 
        print(e)
        print("Look at:" + f_name + " and " + g_name)
	sys.exit()
        return
    
def pickle_population(pop, gen, log, checkpoint_name):
    print("Pickling population from generation %d..." % gen)
    cp = dict(population=pop, generation=gen, logbook=log)

    with open(checkpoint_name, "wb") as cp_file:
        pickle.dump(cp, cp_file)
    return;

def data_summary(logbook):
    print(logbook)
    gen = logbook.select("gen")
    

    avg_accuracy = logbook.chapters["accuracy"].select("avg")
    avg_ce = logbook.chapters["cross_entropy"].select("avg")
    avg_norm = logbook.chapters["norm"].select("avg")
    
    weights = (-1.0, -1.0, 2.0)
    
    fitness = (weights[0]*np.asfarray(avg_ce)) + (weights[1]*np.asfarray(avg_norm)) + (weights[2]*np.asfarray(avg_accuracy))
    
    fig, ax1 = plt.subplots()
    line1 = ax1.plot(gen, avg_accuracy, "b-")
    ax1.set_xlabel("Generation")
    #ax1.set_ylabel("Accuracy", color="b")
    #ax1.xticks(np.arange(0, max(gen)+1, 1.0))
    for tl in ax1.get_yticklabels():
        tl.set_color("b")
    plt.show()
    return

def stats_setup():
    logbook = tools.Logbook()
    #stats_ce = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    #stats_norm = tools.Statistics(key=lambda ind: ind.fitness.values[1])
    stats_accuracy = tools.Statistics(key=lambda ind: ind.fitness.values[0])
    mstats = tools.MultiStatistics(accuracy=stats_accuracy)
    mstats.register("avg", np.mean)
    mstats.register("std", np.std)
    mstats.register("min", np.min)
    mstats.register("max", np.max)
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
