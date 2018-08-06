import multiprocessing
import numpy as np
import pickle

data = range(100)

def square(num):
    return[i**2 for i in num]

def chunk(list, chunk_size):
    return [list[i:i+chunk_size] for i in range(0, len(list), chunk_size)]

pool= multiprocessing.Pool(4)

result = pool.map(square, chunk(data,10))


print(data)
print(chunk(data,10))
print(result)
result = np.asarray(result)
print(np.concatenate(result).ravel().tolist())

def load_data():
    data_filename = 'processed_training_data'
    infile = open(data_filename,'rb')
    test_data = pickle.load(infile)
    print([len(list) for list in test_data])
    return;

load_data()