import sys
#This needs to be streamlined to make code portable
sys.path.append('/localhome/mbaxsej2/optimisation_env/NE15')
print(sys.path)
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
import sys, os
from time import sleep
from spinnman.exceptions import SpinnmanIOException
from spinn_front_end_common.utilities import globals_variables
from elephant.statistics import mean_firing_rate
from numpy import number

home = os.environ['VIRTUAL_ENV']
NE15_path = home + '/git/NE15'

#np.array([(,,)], dtype=[('input', 'i4'),('output', 'i4'), ('weight', 'f4')])
    

class NetworkModel(object):
    '''Class representing model'''
   
    def __init__(self, simtime, timestep, neurons_per_core, input_pop_size, pop_1_size, output_pop_size, on_duration, off_duration, gene=None):
        self.spiketrains = {}
        self.simtime = simtime
        self.timestep = timestep
        self.neurons_per_core = neurons_per_core
        self.input_pop_size = input_pop_size
        self.pop_1_size = pop_1_size
        self.output_pop_size = output_pop_size
        if gene == None:
            print("generating test gene")
            gene = self.generate_test_gene()
        self.gene = gene
        self.weights_1, self.weights_2 = self.gene_to_weights()
        self.cost = None
        self.test_images = None
        self.test_labels = None
        self.test_periods = None
        self.select_random_test_images()
        self.label = None
        self.on_duration = on_duration
        self.off_duration = off_duration   
        self.generate_input_st() 
        self.test_periods = self.generate_test_periods()

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
                    weights_1.append((i, j, self.gene[counter], 0.1))
                counter += 1
        
        for i in range(0, self.pop_1_size):
            for j in range (0, self.output_pop_size):
                if abs(self.gene[counter]) > 0.05:
                    weights_2.append((i, j, self.gene[counter], 0.1))
                counter += 1        
        return weights_1, weights_2;
    
    def test_model(self, num_retries=0):
        '''Testing the model against test data with retry'''
        
        max_retries = 10
        
        def run_sim():            
            print("setting up")
            sim.setup(self.timestep)
            sim.set_number_of_neurons_per_core(sim.IF_curr_exp, self.neurons_per_core)
            sleep(0.5) # such that 
            print("setting up pops")
            input_pop = sim.Population(self.input_pop_size, sim.SpikeSourceArray(self.spiketrains['input_pop']), label="input")
            pop_1 = sim.Population(self.pop_1_size, sim.IF_curr_exp(), label="pop_1")
            output_pop = sim.Population(self.output_pop_size, sim.IF_curr_exp(), label="output_pop")
            print("setting up projs")
            self.input_proj = sim.Projection(input_pop, pop_1, sim.FromListConnector(self.weights_1), 
                                        synapse_type=sim.StaticSynapse())
            self.output_proj = sim.Projection(pop_1, output_pop, sim.FromListConnector(self.weights_2), 
                                        synapse_type=sim.StaticSynapse())
            print("setting up recorders")
            pop_1.record(["spikes"])
            output_pop.record(["spikes"])
            input_pop.record(["spikes"]) 
            
            print("running sim")
            sim.run(self.simtime)
            print("getting data")
            self.spiketrains['input_pop'] = input_pop.get_data(variables=["spikes"]).segments[0].spiketrains
            self.spiketrains['pop_1'] = pop_1.get_data(variables=["spikes"]).segments[0].spiketrains
            self.spiketrains['output_pop'] = output_pop.get_data(variables=["spikes"]).segments[0].spiketrains
            sim.end()
            print("running cost function")
            self.cost_function()
            print("done")
            return;
                  
        try:
            run_sim()
            if self.cost == None:
                raise Exception
        except Exception as e:
                print(e)
                if num_retries < max_retries:
                    num_retries += 1
                    print("Retry %d..." % num_retries)
                    globals_variables.unset_simulator()
                    self.test_model(num_retries)
                    return;
                if num_retries == max_retries:
                    raise Exception
                    return;
    
    def select_random_test_images(self):
        mndata = MNIST(home)
        mndata.gz = True
        images, labels = mndata.load_testing()
        number_test_images = 10
        self.test_images = np.empty((10, 784))
        self.test_labels = np.empty((10))    
        for i in range(0, number_test_images):
            index = random.randrange(0, len(images))
            self.test_images[i] = images[index]
            self.test_labels[i] = labels[index]
        return;
    
    def one_hot_encode_labels(self):
        label_array = np.zeros((len(self.test_labels), 10))
        self.test_labels = np.asarray(self.test_labels, dtype=int)
        label_array[np.arange(10), self.test_labels] = 1
        return label_array;
        
    def generate_test_periods(self):
        number_test= len(self.test_images)
        test_time = self.on_duration + self.off_duration
        test_periods = np.arange(start= 0, step= test_time, stop= test_time*11)
        return test_periods;
    
    def generate_input_st(self):
        '''Generating the poisson spiketrains from image'''
        number_tests = len(self.test_images)

        spikes_all = np.empty((784,), dtype=np.object_)
        spikes_all.fill([])
        for i in range(1,number_tests):
            img = np.reshape(self.test_images[i], [28,28])
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
    
    def cost_function(self):
        '''Returns the value of the cost function for the test.'''
        label_array = self.one_hot_encode_labels()
        spiketrain = self.spiketrains["output_pop"]
        rates = np.zeros([10,10])
        for i in range(0, len(self.test_periods)-1):
            for j in range(0, len(spiketrain)):
                rates[i][j] = mean_firing_rate(spiketrain[j], self.test_periods[i], self.test_periods[i+1])
        
        normalised_rates = rates / rates.sum(axis=0)
        predictions = np.argmax(normalised_rates, axis=0)
        
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
        '''
        #L1 norm (minimise to give sparsity)
        #average weight magnitude
        norm  = 0.0
        for i in self.gene:
            norm += abs(i)
        norm = norm/len(self.gene)
        
        self.cost = (accuracy,)
        
        print(self.cost)
        return;

    def generate_poisson_spiketrains(self, rate):
        '''generates a poissonian test spiketrain'''
        spiketrains = []
        for i in range(0, self.input_pop_size):
            spiketrains.append(poisson_generator(rate, 0, self.simtime))
        return spiketrains;

    
    def visualise_input(self):
        pylab.figure(1)
        spikes = self.spiketrains['input_pop']
        output = []
        for spike in spikes:
            output.append(sum(1 for i in spike))
        output = np.divide(output, float(np.sum(output)))
        output = np.reshape(output,(28,28))
        plt.imshow(output)
        pylab.figure(2)
        plt.imshow(self.img)
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
        super(MnistModel, self).__init__(simtime=2100, timestep=0.1, neurons_per_core=255, input_pop_size=784, pop_1_size=100, output_pop_size=10, on_duration=200, off_duration=10, gene=None)

class ConvMnistModel(NetworkModel):
    '''convolutional MNIST model'''
    number_test_images = 10
    on_duration = 200
    off_duration = 10
    simtime=(on_duration + off_duration)*number_test_images
    timestep=0.1
    neurons_per_core=255
    input_pop_size=784
    filter_size = 5
    image_size = int(input_pop_size**0.5)
    conv_per_line = image_size-filter_size+1 
    pop_1_size = int(conv_per_line**2)
    output_pop_size = 10

    
        
    def __init__(self, gene=None):
        super(ConvMnistModel, self).__init__(self.simtime, self.timestep, self.neurons_per_core, self.input_pop_size, self.pop_1_size, self.output_pop_size, self.on_duration, self.off_duration, gene=None)

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
                        weights_1.append(((top_left_position+(k*self.image_size)+l), neuron_number, gene1[(k*self.filter_size)+l], 0.1))
        counter = 0
      
        for i in range(0, self.pop_1_size):
            for j in range(0, self.output_pop_size):
                weights_2.append((i,j,gene2[counter], 0.1))
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
#testModel = ConvMnistModel(5)

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
#print(testModel.cost)
  



        
#
#raster_plot_spike(out.segments[0].spiketrains)
#pylab.show()
#plt.show()
#print(testModel.weights)

