"""
Abstract Convolutional Neural Network
~~~~~~~~~~~~~~~~~~~~

Yuzhong Huang
2017.5.3

A module to implement an abstract convolutional neural 
network algorithm in a feedforward manner.  Note that it 
is not optimized, and omits many desirable features.

"""

import numpy as np
import operator

class Network(object):
    def __init__(self, input_size, sizes, neuron_nums, connection_nums):
        """
        Initialize an abstract convolutional network where certain 
        number of neurons are randomly distributed on each layer, 
        except input and output layers.  Each layer is a square whose 
        size are indicated by 'sizes'.  Each neuron in hidden layers 
        connects to nearest neurons on the layer below.  Neurons on 
        one layer has same number of connections indicated by 
        connection_nums.

        sizes: a list of ints representing spatial sizes of layers

        neuron_nums: a list of ints representing number of neurons 
        on each layer

        connection_nums: a list of ints representing number of 
        connections each neuron has in each layer

        """
        self.input_size = input_size
        self.sizes = sizes
        self.neuron_nums = neuron_nums
        self.connection_nums = connection_nums

        self.get_input_pos()    # generate a list positions on the input layer
        self.create_neurons()   # populate layers of neurons
        self.init_pos()         # initialize each neuron's x-y position on each layer
        self.form_connection()  # connect neurons to form a network

    def get_input_pos(self):
        """
        Create a list of x-y coordinates and stored it to self.input_poses

        """
        self.input_poses = []

        for i in range(self.input_size):
            for j in range(self.input_size):
                self.input_poses.append((i,j))

    def create_neurons(self):
        """
        Create a list of lists of Neuron objects and assign it to 
        self.neurons.  The first list represents a list of layers, 
        the second lists represents a list of neurons in each layer

        """
        self.neurons = []

        # loop through layers
        for i in range(len(self.neuron_nums)):
            neurons = []    # neuron list for a layer
            for j in range(self.neuron_nums[i]):    # append #(self.neuron_nums[i]) Neuron objects to ``neurons``
                neurons.append(Neuron(self.connection_nums[i])) 
            self.neurons.append(neurons)

    def init_pos(self):
        """
        Randomly populate the each neuron's spatial position on each layer

        """
        self.poses = []

        # populate random sorted x-y coordinates for each layer
        for i in range(len(self.sizes)):
            # randomly generate x-y coordinates
            offset = int((self.input_size-self.sizes[i])/2)
            xs = np.random.uniform(low=offset, high=self.input_size-offset-1, size=(self.neuron_nums[i],))
            ys = np.random.uniform(low=offset, high=self.input_size-offset-1, size=(self.neuron_nums[i],))
            pos = [(x,y) for x, y in zip(xs, ys)]

            # sort the list based on x value
            pos.sort(key=lambda x:x[0]) 
            self.poses.append(pos)

        # assign poses to neurons
        for i in range(len(self.neuron_nums)):
            for j in range(self.neuron_nums[i]):
                self.neurons[i][j].set_pos(self.poses[i][j])

    def form_connection(self):
        """
        Connect neurons to form a network. Each neuron connects to 
        nearest output neurons in the previous layer. Connections are
        stored in self.connections, where the first list represents 
        each layer, the second list represents each neuron, the third 
        list contains indices of neurons to connect in the previous 
        layer. Note that for the first layer, the indices refers to 
        input signals in the input layer

        """
        self.connections = []

        for i in range(len(self.neurons)):
            connections = []
            for j in range(len(self.neurons[i])):
                connections.append(self.get_connections(self.neurons[i][j].pos, i, self.connection_nums[i]))
            self.connections.append(connections)

    def get_connections(self, pos, layer, num):
        """
        Get a list of indices of neurons to connect given the position
        and layer of a neuron
        """  
        # copy the list of positions in the previous layer if the neuron is on 
        # the first layer, get the positions of the input layer     
        if layer == 0:
            poses = self.input_poses[:]
        else:
            poses = self.poses[layer-1][:]

        # subtract pos from each positions in the list to get a distance vector
        for i in range(len(poses)):
            poses[i] = tuple(map(operator.sub, poses[i], pos))

        # sort the indices based on distance vectors
        indices = list(range(len(poses)))
        indices.sort(key=lambda x: poses[x][0]**2+poses[x][1]**2)

        return indices[:num]

    def feedforward(self, a):
        """
        Feedforward an input to the neural network to get a result

        """
        for i in range(len(self.neurons)):
            b = np.zeros((len(self.neurons[i]),))
            for j in range(len(self.neurons[i])):
                b[j] = self.neurons[i][j].forward(np.take(a, self.connections[i][j]))
            a = b
        return a

    def backprop(self, a, y, eta):
        """
        Backprop an error to the network to tune the weights

        """
        activations = []
        for i in range(len(self.neurons)):
            b = np.zeros((len(self.neurons[i]),))
            for j in range(len(self.neurons[i])):
                b[j] = self.neurons[i][j].forward(np.take(a, self.connections[i][j]))
            a = b
            activations.append(a)

        delta = self.cost_derivative(activations[-1], y) * \
            relu_prime(zs[-1])

        for i in range(len(self.neurons), -1, -1):            
            for j in range(len(self.neurons[i])):
                self.neurons[i][j].backward(delta, eta)
            activation = activations[i]
            sp = self.relu_prime(activation)
            delta *= sp

    def cost_derivative(self, output_activations, y):
        """Return the vector of partial derivatives \partial C_x /
        \partial a for the output activations."""
        return (output_activations-y) 

    def relu_prime(self, z):
        return 0 if z<=0 else 1

class Neuron(object):
    def __init__(self, num):
        """
        Build a simple neuron with a bias and a array of weights

        num: number of weights

        """
        self.weights = np.random.randn(num)
        self.bias = np.random.randn()

    def set_pos(self, pos):
        """
        Set neuron's x-y coordinates

        pos: (x, y)
        """
        self.pos = pos

    def forward(self, inputs):
        """
        Generate a single output from multiplying weights and inputs

        inputs: a vector of neuron signals

        """
        res = np.dot(self.weights, inputs) + self.bias
        return res if res>0 else 0  #relu

    def backward(self, error, rate):
        """
        Tune the weights based on error and rate

        """
        self.bias -= rate * error
        self.weights -= self.weights * error