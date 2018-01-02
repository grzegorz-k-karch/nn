import numpy as np
import random
import mnist_loader as mloader

## keep training separated from network

class CrossEntropyCost(object):

    @staticmethod
    def fn(a, y):
        return 0.5*np.linalg.norm(a-y)**2

class SGD(object):

    def __init__(self, a_fn, num_epochs, batch_size, learning_rate, cost_fn):

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cost_fn = cost_fn
        self.a_fn = a_fn
        self.done = 0
        
    def train(self, network, training_data):

        for i in range(self.num_epochs):

            random.shuffle(training_data)
            num_batches = len(training_data)//self.batch_size
            
            for j in range(num_batches):

                start = j*self.batch_size
                end = (j+1)*self.batch_size
                end = min(end, len(training_data))

                self.update_batch(network, training_data[start:end])

    def update_batch(self, network, batch):

        batch_dw = [np.zeros(w.shape) for w in network.weights]
        batch_db = [np.zeros(b.shape) for b in network.biases]

        for sample in batch:

            a = self.prop(network, sample[0])
            y = sample[1]

            cost = self.cost_fn.fn(a[-1], y)
            sample_dw, sample_db = self.backprop(network, y - a[-1], a)
#             delta_w += sample_dw
#             delta_b += sample_db


        network.weights = [w - self.learning_rate/len(batch)*dw for w, dw in zip(network.weights, batch_dw)]
        network.biases = [b - self.learning_rate/len(batch)*db for b, db in zip(network.biases, batch_db)]



    def prop(self, network, x):

        a = [] # activations
        a.append(x)
        for l in xrange(network.num_layers - 1):
            z = (network.weights[l]).dot(a[l]) + network.biases[l]
            a.append(network.activation_fn(z))

        return a

    def backprop(self, network, d, a):

        dw = [np.zeros(w.shape) for w in network.weights]
        db = [np.zeros(b.shape) for b in network.biases]
        # 1
        t1 = d*self.a_fn_prime(a[1])
        db[1] = t1
        dw[1] = a[1]*t1
        # 2
        
        return 1,2

    def a_fn_prime(z):

        a = self.a_fn(z)
        return a*(1 - a)


class Network1(object):
    
    def __init__(self, graph, graph_initialization=None):

        # [784, 30, 10]
        self.graph = graph
        self.num_layers = len(graph)
        
        if graph_initialization is not None:

            self.weights, self.biases = graph_initialization(graph)
            
        else:

            self.normalized_weights()

    def normalized_weights(self):

        self.biases = [np.random.randn(y, 1) for y in self.graph[1:]] 
        self.weights = [np.random.randn(y, x)/np.sqrt(x) \
                        for x, y in zip(self.graph[:-1], self.graph[1:])]

    def activation_fn(self, z):

        return 1/(1 + np.exp(-z))
    
    
                       
training_data, validation_data, test_data = mloader.load_data("data/mnist.pkl.gz")

network = Network1([784, 30, 10])

# training = SGD(num_epochs=30, batch_size=50, learning_rate=0.1, cost_fn=CrossEntropyCost)
training = SGD(a_fn=network.activation_fn, num_epochs=1, batch_size=50, learning_rate=0.1, cost_fn=CrossEntropyCost)

training.train(network, training_data[:100])
