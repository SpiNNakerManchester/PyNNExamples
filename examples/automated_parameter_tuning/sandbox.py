import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from mnist import MNIST
import os.path
import numpy as np
from Crypto.SelfTest.Cipher.test_AES import test_data
import random
home = os.path.expanduser("~")



#one of each digit [number 0s, number 1s ...]





generate_test_images()
plt.imshow(np.reshape(np.asarray(test_images[1][1]),(28,28)))
plt.show()

# old code 

    
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