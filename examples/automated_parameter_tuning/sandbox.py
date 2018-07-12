import numpy as np
import matplotlib.pyplot as plt
from time import sleep
image_size = 28
filter_size = 5
fig = plt.gcf()

test_input_matrix = np.zeros((28,28))
    
conv_per_line = image_size-filter_size +1

'''
for i in range (20, conv_per_line):
    #rows
    for j in range(0, conv_per_line):
        #columns
        print((i*conv_per_line)+j)
        for k in range(0, filter_size):
            for l in range(0, filter_size):
                test_input_matrix[i+k,j+l] = 1       
        plt.imshow(test_input_matrix)
        fig.show()
        fig.canvas.draw()
        test_input_matrix = np.zeros((28,28))
        sleep(0.001)
    
'''
gene1 = np.random.rand(576)
weights_1 = []
for i in range (0, conv_per_line):
    #image row index
    for j in range(0, conv_per_line):
        #image column index
        neuron_number = (i*conv_per_line) + j
        top_left_position = (i*image_size) + j
        for k in range(0, filter_size):
            #filter row index
            for l in range(0, filter_size):
                #filter column index
                # (input image, output neurons, weight matrix)
                weights_1.append(((top_left_position+(k*image_size)+l), neuron_number, gene1[(k*filter_size)+l], 0.1))
                
print weights_1[-1:]