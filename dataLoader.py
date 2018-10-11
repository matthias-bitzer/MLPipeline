import numpy as np
import log

class DataLoader(object):

    def __init__(self):
        self.counter = 0
        self.load_data()


    def load_data(self):
        from tensorflow.examples.tutorials.mnist import input_data
        mnist = input_data.read_data_sets('MNIST_data/',one_hot=True)
        self.data = mnist.train.images
        self.labels = mnist.train.labels
        self.train_size = np.shape(self.data)[0]


    def get_batch(self,batch_size):
        Images = self.data[self.counter:self.counter+batch_size,:]
        labels = self.labels[self.counter:self.counter+batch_size,:]
        images = np.array(Images)
        images=images.reshape(batch_size,28,28,1)

        if self.counter + batch_size + 1 <= (self.train_size-batch_size):
            self.counter = self.counter + batch_size + 1
        else:
            self.counter=0

        return images,labels