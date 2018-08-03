import multiprocessing
import numpy as np

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