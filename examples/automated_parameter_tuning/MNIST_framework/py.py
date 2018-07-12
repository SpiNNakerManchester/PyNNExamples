from mnist import MNIST

class MNIST_model():
    ''' A SpiNNaker SNN model for MNIST character recognition task.'''
        
    def __init__(self, size, weights):
        self.size = size
        self.weights = weights
        self.training_accuracy= 0
        self.testing_accuracy = 0
        self.is_trained = False
        
    def train_model(self):
        images, labels = mndata.load_training()
        for i in images:
            self.training_accuracy = 0
            #generate st and present with label to network multiple times
        self.is_trained=True

    def test_model(self):
        if not(self.is_trained):
            self.train_model()
        images, labels = mndata.load_testing()
        for i in images:
            self.testing_accuracy = 0
            #test and calc accuracy
            
mndata = MNIST('./dir_with_mnist_data_files')
mndata.gz = True

for i in range (0,9):
    d["model{0}".format(i)]= MNIST_model()

