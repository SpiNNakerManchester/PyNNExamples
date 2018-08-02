import sys, os
import copy
from neo.core import Segment, SpikeTrain
from quantities import s, ms
from duplicity.globals import num_retries
#Dependencies need to be sorted
#sys.path.append('/localhome/mbaxsej2/optimisation_env/NE15')
#home = os.path.expanduser("~")
home = os.environ['VIRTUAL_ENV']
NE15_path = home + '/git/NE15'
sys.path.append(NE15_path)
#This needs to be streamlined to make code portable
import traceback
from decimal import *
import spynnaker8 as sim
import pickle
import math
import poisson as poisson
import pylab
import numpy
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
import random
from poisson.poisson_tools import poisson_generator
from mnist import MNIST
from os.path import expanduser
from time import sleep
from spinnman.exceptions import SpinnmanIOException
from spinn_front_end_common.utilities import globals_variables
from elephant.statistics import mean_firing_rate, instantaneous_rate
from numpy import number
import multiprocessing
import gc
from spalloc.job import JobDestroyedError
import pprint

def pool_init(l):  
    gc.collect()
    global lock
    lock = l 

def evalModel(gene, gen):
    '''evaluates the model'''
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
        model = ConvMnistModel(gene, gen)
        model.test_model()
        sys.stdout = old_stdout
        sys.stderr = old_stderr            
        print ("Process " + current.name + " finished sucessfully: %s" % (model.cost,)) 
        return model.cost;
      
    #except JobDestroyedError as e:
    #    traceback.print_exc()
	#print(e)
	#raise e
	#sys.exit()    
    except KeyboardInterrupt:
        raise KeyboardInterruptError()
    except Exception as e:
        sys.stdout = old_stdout
        sys.stderr = old_stderr
        print ("Process " + current.name + " stopped unexpectedly.") 
        print(e)
        print("Look at:" + f_name + " and " + g_name)
        sys.exit()
        return
    
    
def evalPopulation(popgen):
    '''evaluates a population of individuals'''
    popgen = np.asarray(popgen)
    pop = popgen[0][:]
    gen = popgen[1]
    
    if len(pop)< 1:
        return;
    
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

    def eval(num_retries=0):
        
        max_retries = 4
        if num_retries < max_retries:
            try:
                print("setting up canonicalModel")
                canonicalModel = ConvMnistModel(pop[0], True, gen)
                canonicalModel.set_up_sim()
                canonicalModel.test_model()
                
                models_dict = {}
                fitnesses = []
                models_dict[0] = canonicalModel
                
                for i in range(1,len(pop)):
                    models_dict[i] = copy.copy(canonicalModel)
                    models_dict[i].gene = pop[i]
                    models_dict[i].weights_1, models_dict[i].weights_2 = models_dict[i].gene_to_weights()
                    models_dict[i].spiketrains = copy.copy(canonicalModel.spiketrains)
                    models_dict[i].test_model()
            
                sim.run(canonicalModel.simtime)
                
                for i in range(0, len(pop)):
                    print(i)
                    models_dict[i].get_sim_data()
            
                sim.end()   
            
                for i in range(0, len(pop)):
                    models_dict[i].cost_function()
                    fitnesses.extend(models_dict[i].cost)
                    
                return fitnesses;
            except Exception:
                eval(num_retries+1)
                return;
        else:
            raise Exception('eval() reached maximum number of retries')
            return;
    
    fitnesses = eval()
    sys.stdout = old_stdout
    sys.stderr = old_stderr          
    print ("Process " + current.name + " finished sucessfully, average accuracy:: %s" % np.average(np.asarray(fitnesses)))
    return fitnesses;
    
class NetworkModel(object):
    '''Class representing model'''
   
    def __init__(self, timestep, neurons_per_core, input_pop_size, pop_1_size, output_pop_size, on_duration, off_duration, gene=None, exsim=False, gen=0):
        self.spiketrains = {}
        self.timestep = timestep
        self.neurons_per_core = neurons_per_core
        self.input_pop_size = input_pop_size
        self.pop_1_size = pop_1_size
        self.output_pop_size = output_pop_size
        if gene == None:
            print("generating test gene")
            gene = self.generate_test_gene()
    	else:
    	    print("gene found")
	    
        self.gene = gene
        print("converting gene to weights")
        self.weights_1, self.weights_2 = self.gene_to_weights()
        self.on_duration = on_duration
        self.off_duration = off_duration
        self.test_set = [4,4,4,4,4,4,4,4,4,4]
        self.number_digits = len(self.test_set)
        self.number_tests = sum(self.test_set)
        self.simtime = (self.on_duration + self.off_duration)*self.number_tests
        self.gen = gen
        self.test_images = None
        self.test_labels = None
        self.test_periods = None
        self.cost = None
        
    	print("generating test data")
        self.generate_test_data()
    	print("generating input spiketrain")
        self.generate_input_st()
        self.exsim = exsim 
        if not self.exsim:
            self.set_up_sim()
            
        self.input_pop = None    
        self.pop_1 = None
        self.ouput = None

        
    def set_up_sim(self):
        print("setting up simulator")
        sim.setup(self.timestep)
        sim.set_number_of_neurons_per_core(sim.IF_curr_exp, self.neurons_per_core)
        self.input_pop = sim.Population(self.input_pop_size, sim.SpikeSourceArray(self.spiketrains['input_pop']), label="input")
        return;
    
    def set_up_model(self):
        print("setting up pops")
        self.pop_1 = sim.Population(self.pop_1_size, sim.IF_curr_exp(), label="pop_1")
        self.output_pop = sim.Population(self.output_pop_size, sim.IF_curr_exp(), label="output_pop")
        print("setting up projs")
        self.input_proj = sim.Projection(self.input_pop, self.pop_1, sim.FromListConnector(self.weights_1), 
                                    synapse_type=sim.StaticSynapse())
        self.output_proj = sim.Projection(self.pop_1, self.output_pop, sim.FromListConnector(self.weights_2), 
                                    synapse_type=sim.StaticSynapse())
        print("setting up recorders")
        self.pop_1.record(["spikes"])
        self.output_pop.record(["spikes"])
        self.input_pop.record(["spikes"])
        return;
    
    def get_sim_data(self):            
        print("getting data")
        self.spiketrains['input_pop'] = self.input_pop.get_data(variables=["spikes"]).segments[0].spiketrains
        self.spiketrains['pop_1'] = self.pop_1.get_data(variables=["spikes"]).segments[0].spiketrains
        self.spiketrains['output_pop'] = self.output_pop.get_data(variables=["spikes"]).segments[0].spiketrains
        return;
    
        
        
    def generate_test_data(self):
        #picking test images from data in accordance with test_set  
        mndata = MNIST(home)
        mndata.gz = True
        images, labels = mndata.load_testing()
        
        test_data = [[] for i in range(self.number_digits)]
        

        for label, image in zip(labels, images):
            test_data[label].append(image)
            
        test_images = [[] for i in range(self.number_digits)]
        test_labels = []
        
        for i in range(len(self.test_set)):
            for j in range(self.test_set[i]):
                #pick = random.randint(0,len(test_data[i])-1)
                length = len(test_data[i])
                pick = (self.gen+length) % length
                picked_image = test_data[i][pick]
                test_images[i].append(picked_image)
                test_labels.append(i)
        self.test_images = test_images
        self.test_labels = np.asarray(test_labels)
        
        # generating the time periods to identify when input presented in the spike train
        
        test_time = self.on_duration + self.off_duration
        self.test_periods = np.arange(start= 0, step= test_time, stop= test_time*(self.number_tests+1))
        #print(self.test_periods)       
        return;
    
    def one_hot_encode_labels(self):
        label_array = np.zeros((self.number_tests, self.number_digits))
        cumulative_test = np.cumsum(self.test_set)
        
        counter = 0
        for i in range(self.number_digits):
            for j in range(self.test_set[i]):
                label_array[counter][i] = 1
                counter += 1
        label_array = np.asarray(label_array, dtype=int)
        return label_array;
    
    def generate_input_st(self):
        '''Generating the poisson spiketrains from image'''
        linear_test_images = []
        for i in self.test_images:
            for j in i:
                linear_test_images.append(j)
        
        spikes_all = np.empty((784,), dtype=np.object_)
        spikes_all.fill([])
        for i in range(self.number_tests):
            img = np.reshape(linear_test_images[i], (28,28))
            img = np.divide(img, 255.0)
            height, width = img.shape
            max_freq = 1000 #Hz
            spikes = poisson.mnist_poisson_gen(numpy.array([img.reshape(height*width)]),\
                                                height, width, max_freq, self.on_duration, self.off_duration)
            spikes = np.asarray(spikes)
            for j in range(0, len(spikes)):
                spikes[j] = [x+(self.on_duration*i)+(self.off_duration*i) for x in spikes[j]]
                spikes_all[j] = spikes_all[j] + spikes[j]
        self.spiketrains['input_pop'] = spikes_all        
        
        return;

    def generate_test_gene(self):
        '''generates a test gene'''
        test_gene =[]
        for i in range(0, self.input_pop_size):
            for j in range(0, self.pop_1_size):
                test_gene.append(random.uniform(-10,10))
        for i in range(0, self.pop_1_size):
            for j in range(0, self.output_pop_size):
                test_gene.append(random.uniform(-10,10))

        return test_gene
    
    def gene_to_weights(self):
        '''converts a gene list to the weights'''      
        weights_1 = []
        weights_2 = []
        counter = 0
        
        for i in range(0, self.input_pop_size):
            for j in range (0, self.pop_1_size):
                if abs(self.gene[counter]) > 0.05:
                    weights_1.append((i, j, self.gene[counter], 1.0))
                counter += 1
        
        for i in range(0, self.pop_1_size):
            for j in range (0, self.output_pop_size):
                if abs(self.gene[counter]) > 0.05:
                    weights_2.append((i, j, self.gene[counter], 1.0))
                counter += 1        
        return weights_1, weights_2;
    
    def test_model(self, num_retries=0):
        '''Testing the model against test data with retry'''
        
        max_retries = 10
        
        def run_sim():
            print("running sim")
            sim.run(self.simtime)
            return;
        
        try:
            self.set_up_model()
            
            if not self.exsim:
                run_sim()
                self.get_sim_data()
                sim.end()
                print("running cost function")
                self.cost_function()
                if self.cost == None:
                    raise Exception
          
        except Exception as e:
            print(e)
            if num_retries < max_retries:
                num_retries += 1
                sleep(20)
                print("Retry %d..." % num_retries)
                globals_variables.unset_simulator()
                self.test_model(num_retries)
                return;
            if num_retries == max_retries:
                raise Exception
        return;
    
    
    
    def cost_function(self):
        '''Returns the value of the cost function for the test.'''
        label_array = self.one_hot_encode_labels()
        spiketrain = self.spiketrains["output_pop"]
        rates = np.zeros((self.number_tests, self.number_digits))
        for i in range(len(self.test_periods)-1):
            for j in range(len(spiketrain)):
                rates[i][j] = mean_firing_rate(spiketrain[j], self.test_periods[i], self.test_periods[i+1])
        
        
        normalised_rates = np.divide(rates, rates.sum(axis=1)[:, None])
        
        #print(normalised_rates)
        
        predictions = np.argmax(normalised_rates, axis=1)

        #print(predictions)
        #print(self.test_labels)
        
        accuracy_array = np.array(predictions == self.test_labels)
        accuracy = np.true_divide(np.sum(accuracy_array),len(accuracy_array))         
        
        '''
        output = np.divide(spikes, float(np.sum(spikes)))
        
        #Cross entropy
        cross_entropy = 0
        for i in range (0,len(output)):
            if output[i]>0.0:
                cross_entropy += label_list[i]*math.log(output[i])/math.log(2)
        cross_entropy = cross_entropy * -1

        #L1 norm (minimise to give sparsity)
        #average weight magnitude
        norm  = 0.0
        for i in self.gene:
            norm += abs(i)
        norm = norm/len(self.gene)
        '''
        
        self.cost = (accuracy,)
        return;

    def generate_poisson_spiketrains(self, rate):
        '''generates a poissonian test spiketrain'''
        spiketrains = []
        for i in range(0, self.input_pop_size):
            spiketrains.append(poisson_generator(rate, 0, self.simtime))
        return spiketrains;

    
    def visualise_input(self):
        
        #must be adapted for variable length input (train set other than all ones)
        
        input_spiketrain = self.spiketrains['input_pop']
        input_neo = Segment()
        
        for i in input_spiketrain:
            input_neo.spiketrains.append(SpikeTrain(times=i*ms, t_stop=2100))
       
        mean_firing_rates = []
        
        for st in input_neo.spiketrains:
            mean_firing_rates.append(float(mean_firing_rate(st, self.test_periods[9], self.test_periods[10])))
        print mean_firing_rates
        
        inimage = np.reshape(np.asarray(mean_firing_rates), (28,28))
        plt.imshow(inimage)
        plt.show()
        
        
        return;

    
    def visualise_input_weights(self):
        
        def do_plot(ax, img):
            im = ax.imshow(img)
            return;
        
        ax_list = range(0, self.pop_1_size)
        ax_list = ["ax" + str(s) for s in ax_list]
        fig, ax_list = plt.subplots(1, len(ax_list))
        for neuron in range(self.pop_1_size):
            weight_image =[]
            for i in range(0, (self.pop_1_size*self.input_pop_size), self.pop_1_size):
                weight_image.append(self.gene[i + neuron])
            weight_image = np.reshape(weight_image, (28,28))
            weight_image *= 255.0/weight_image.max()
            do_plot(ax_list[neuron], weight_image)
        plt.show()
        return

    def visualise_output_weights(self):
        #broken
        def do_plot(ax, img):
            im = ax.imshow(img)
            return;
        
        ax_list = range(0, self.output_pop_size)
        ax_list = ["ax" + str(s) for s in ax_list]
        fig, ax_list = plt.subplots(1, len(ax_list))
        for neuron in range(self.output_pop_size):
            weight_image =[]
            for i in range(self.input_pop_size*self.pop_1_size, len(self.gene), self.output_pop_size):
                weight_image.append(self.gene[i + neuron])
            weight_image = np.array(weight_image)
            weight_image *= 255.0/weight_image.max()
            do_plot(ax_list[neuron], weight_image)
        plt.show()
        return;
    
    def plot_spiketrains(segment):
        for spiketrain in segment.spiketrains:
            y = np.ones_like(spiketrain) * spiketrain.annotations['source_id']
            plt.plot(spiketrain, y, '.')
            plt.ylabel(segment.name)
            plt.setp(plt.gca().get_xticklabels(), visible=False)
        return; 

class MnistModel(NetworkModel):
    def __init__(self, gene=None):
        super(MnistModel, self).__init__(timestep=1, neurons_per_core=255, input_pop_size=784, pop_1_size=100, output_pop_size=10, on_duration=200, off_duration=10, gene=None)

class ConvMnistModel(NetworkModel):
    '''convolutional MNIST model'''
    on_duration = 200
    off_duration = 10
    timestep=1.0
    neurons_per_core=255
    input_pop_size=784
    filter_size = 5
    image_size = int(input_pop_size**0.5)
    conv_per_line = image_size-filter_size+1 
    pop_1_size = int(conv_per_line**2)
    output_pop_size = 10
           
    def __init__(self, gene=None, exsim=False, gen=0):
        super(ConvMnistModel, self).__init__(self.timestep, self.neurons_per_core, self.input_pop_size, self.pop_1_size, self.output_pop_size, self.on_duration, self.off_duration, gene, exsim, gen)

    def gene_to_weights(self):
        '''converts gene to convolutional weights in gene1 and fully connected gene2'''
        filter_square_size = self.filter_size**2
        
        gene1 = self.gene[:filter_square_size]
        gene2 = self.gene[filter_square_size:]
        
        weights_1 = []
        weights_2 = []
        
        #This is difficult to visualise
        
        for i in range (0, self.conv_per_line):
            #image row index
            for j in range(0, self.conv_per_line):
                #image column index
                neuron_number = (i*self.conv_per_line) + j
                top_left_position = (i*self.image_size) + j
                for k in range(0, self.filter_size):
                    #filter row index
                    for l in range(0, self.filter_size):
                        #filter column index
                        # (input image, output neurons, weight matrix)
                        weights_1.append(((top_left_position+(k*self.image_size)+l), neuron_number, gene1[(k*self.filter_size)+l], 1))
        counter = 0
      
        for i in range(0, self.pop_1_size):
            for j in range(0, self.output_pop_size):
                weights_2.append((i,j,gene2[counter], 1))
                counter += 1
                
        return weights_1, weights_2
    
    def generate_test_gene(self):
        '''generates a test gene'''
        test_gene =[]
        for i in range(0, self.filter_size**2):
            test_gene.append(random.uniform(-10,10))
        for i in range(0, self.pop_1_size):
            for j in range(0, self.output_pop_size):
                test_gene.append(random.uniform(-10,10))
        
        return test_gene

    def visualise_filter(self):
        filter_image = np.reshape(gene[:filter^2], (28,28))
        pylab.figure(3)
        plt.imshow(filter_image)
        plt.show()
            
        
        
# Test code
#Conv test code
#testModel = ConvMnistModel()
#print(testModel.test_labels)

#testModel.visualise_input()
#testModel = MnistModel()
#testModel.spiketrains = pickle.load( open( "testModelstsave.p", "rb" ) )
#print(testModel.spiketrains["output_pop"])
#testModel.one_hot_encode_labels()
#testModel.generate_test_periods()
#testModel.test_model()
#pickle.dump(testModel.spiketrains, open( "testModelstsave.p", "wb" ) )

#testModel.visualise_input()
#testModel.visualise_input_weights()
#testModel.visualise_output_weights()
#testModel.cost_function()
  



        
#
#raster_plot_spike(out.segments[0].spiketrains)
#pylab.show()
#plt.show()
#print(testModel.weights)

