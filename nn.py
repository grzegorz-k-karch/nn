import numpy as np
import random
import mnist_loader

## keep training separated from network

#cost
class mse(object):

    @staticmethod
    def eval(y, a):
        return 0.5*np.linalg.norm(a-y)**2

    @staticmethod
    def eval_prime(y, a):
        return y - a


#model
class mlp(object):

    #-------------------------------------------------------------------------
    def __init__(self, graph, graph_initialization=None):

        # [784, 30, 10]
        self.graph = graph
        self.num_layers = len(graph)

        if graph_initialization is not None:
            self.weights, self.biases = graph_initialization(graph)
        else:
            self.normalized_weights()

    #-------------------------------------------------------------------------
    def normalized_weights(self):

        self.biases = [np.random.randn(y, 1) for y in self.graph[1:]]
        self.weights = [np.random.randn(y, x)/np.sqrt(x) \
                        for x, y in zip(self.graph[:-1], self.graph[1:])]

    #-------------------------------------------------------------------------
    def activation_fn(self, z):

        return 1/(1 + np.exp(-z))

    #-------------------------------------------------------------------------
    def a_fn_prime(self, z):

        a = self.activation_fn(z)
        return a*(1 - a)

    #-------------------------------------------------------------------------
    def feed_forward(self, sample, zs=None):

        x = sample[0]
        a = x[:]
        if zs is not None:
            zs.append(a)

        for l in range(self.num_layers - 1):
            z = (self.weights[l]).dot(a) + self.biases[l]
            a = self.activation_fn(z)
            if zs is not None:
                zs.append(z)

        return a

    #-------------------------------------------------------------------------
    def feed_backward(self, cost_prime, zs):

        d = cost_prime
        dw = [np.zeros(w.shape) for w in self.weights]
        db = [np.zeros(b.shape) for b in self.biases]
        
        for l in range(self.num_layers - 1):
            # pass activation function
            # http://neuralnetworksanddeeplearning.com/chap2.html <- BP1
            print(str(len(d)) + " <> " + str(len(self.a_fn_prime(zs[-1-l]))))

            d = d*self.a_fn_prime(zs[-1-l]) # element-wise multiplication
            db[-1-l] = d
            dw[-1-l] = zs[-1-l-1].dot(d.transpose())
            # propagate the error further
            d = ((self.weights[-1-l]).transpose()).dot(d)

            
            print(str(len(d)) + " < ")
        

# dataset
training_data, validation_data, test_data = mnist_loader.load_data("/home/gkk/data/mnist.pkl.gz")

network = mlp([784, 30, 10])

zs = []

output = network.feed_forward(training_data[0], zs)
print("output:\n")
print(output)
print("\n")

print("zs:\n")
print(zs)
print("\n")

digits = [x for x in range(10)]

ref_output = training_data[0][1]
print("ref output %s\n" % digits[np.argmax(ref_output)])
print(ref_output)
print("\n")

cost_fn = mse()
cost = cost_fn.eval(ref_output, output)
print("cost\n")
print(cost)
print("\n")

network.feed_backward(cost_fn.eval_prime(ref_output, output), zs)


# # training = sgd(num_epochs=30, batch_size=50, learning_rate=0.1, cost_fn=mse)
# training = sgd(a_fn=network.activation_fn, num_epochs=1, batch_size=50, learning_rate=0.1, cost_fn=mse)


# training.train(network, training_data[:500])

# print(training.test(network, test_data))


#optimizer
class sgd(object):

    #-------------------------------------------------------------------------
    def __init__(self, a_fn, num_epochs, batch_size, learning_rate, cost_fn):

        self.num_epochs = num_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.cost_fn = cost_fn
        self.a_fn = a_fn
        self.done = 0

    #-------------------------------------------------------------------------
    def train(self, network, training_data):

        for i in range(self.num_epochs):

            random.shuffle(training_data)
            num_batches = len(training_data)//self.batch_size

            for j in range(num_batches):

                start = j*self.batch_size
                end = (j+1)*self.batch_size
                end = min(end, len(training_data))

                self.update_batch(network, training_data[start:end])

    #-------------------------------------------------------------------------
    def update_batch(self, network, batch):

        batch_dw = [np.zeros(w.shape) for w in network.weights]
        batch_db = [np.zeros(b.shape) for b in network.biases]

        cost = 0.0

        for sample in batch:

            yp, zs = self.prop(network, sample[0])
            y = sample[1]

            cost += self.cost_fn.fn(yp, y)

            sample_dw, sample_db = self.backprop(network, y - yp, zs)
            batch_dw = [bdw + sdw for bdw, sdw in zip(batch_dw, sample_dw)]
            batch_db = [bdb + sdb for bdb, sdb in zip(batch_db, sample_db)]

        network.weights = [w - self.learning_rate/len(batch)*dw for w, dw in zip(network.weights, batch_dw)]
        network.biases = [b - self.learning_rate/len(batch)*db for b, db in zip(network.biases, batch_db)]

    #-------------------------------------------------------------------------
    def prop(self, network, x):

        zs = []
        a = x[:]
        for l in xrange(network.num_layers - 1):
            z = (network.weights[l]).dot(a) + network.biases[l]
            zs.append(z)
            a = network.activation_fn(z)

        # a is the network output, zs are the layer inputs
        return a, zs

    #-------------------------------------------------------------------------
    def backprop(self, network, d, zs):

        s = d

        dw = [np.zeros(w.shape) for w in network.weights]
        db = [np.zeros(b.shape) for b in network.biases]
        # 1
        a = s*self.a_fn_prime(zs[1])
        db[1] = a # bias just gets the value
        dw[1] = network.weights[1]*a

        s = network.weights[1].transpose().dot(a)

        # 2
        a = s*self.a_fn_prime(zs[0])
        db[0] = a
        dw[0] = network.weights[0]*a

        return dw, db

    #-------------------------------------------------------------------------
    def test(self, network, test_data):


        cost = 0.0
        for i in range(len(test_data)):

            sample = test_data[i]

            yp, _ = self.prop(network, sample[0])
            cost += self.cost_fn.fn(yp, sample[1])

        return cost/len(test_data)
