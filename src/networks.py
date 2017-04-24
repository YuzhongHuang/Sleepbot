"""
Async Neural Network
~~~~~~~~~~~~~~~~~~~~

Yuzhong Huang
2017.4.11

A module to implement an async neural network algorithm 
in a feedforward manner.  Individual neurons has a nucleus 
and only fires when accumlative nucleus is beyond certain 
threshold value.  Note that it is not optimized,
and omits many desirable features.
"""

import numpy as np

class Network(object):
    def __init__(self, sizes, density, num):
        """ 
        The list ``sizes`` contains the unit lengths of respective 
        dimensions of the network.  The first two dimensions are 
        consistent with the input image. 

        The integer ``density`` refers to the number of output 
        connections that each neuron has. 

        The integer ``num`` indicates number of neurons in the 
        network's hidden layer.  Note that there are #(img pixels) 
        input neurons in addition to the hidden layer """

        self.sizes = sizes
        self.density = density
        self.num = num
        self.output = []

        #(input layer + hidden layer) of neurons
        self.neurons = [Neuron() for i in range(sizes[0]*sizes[1] + num)] 

        # initialize signals that store all the intermediate inputs 
        # to individual neurons, including input neurons
        self.signals = np.zeros(len(self.neurons))

        # initialize neuron's positions and form connections
        self.pos = []
        self.init_pos()
        self.form_connection()


    def init_pos(self):
        """
        Randomly populate the neurons' 3D positions in the space 
        given by self.sizes """

        # input layer neurons have z values equal to 0
        for i in range(self.sizes[0]):
            for j in range(self.sizes[1]):
                self.pos.append((i,j,0))

        # randomly generate hidden layer neurons
        xs = np.random.uniform(low=0, high=self.sizes[0], size=(self.num,))
        ys = np.random.uniform(low=0, high=self.sizes[1], size=(self.num,))
        zs = np.random.uniform(low=0, high=self.sizes[2], size=(self.num,))
        pos = [(x,y,z) for x, y, z in zip(xs, ys, zs)]

        # sort the list based on z value
        pos.sort(key=lambda x:x[2]) 
        self.pos += pos

    def form_connection(self):
        """
        Connects neurons to form a network. Each neuron connects to 
        #density of nearest output neurons that has z vaule higher 
        than itself """

        # loop through all the neurons spatially to form connections
        for i in range(len(self.pos)):
            self.neurons[i].connect(self.find_connection(self.pos[i], i))

    def find_connection(self, pos, i):
        """
        Find indices of #density of nearest output neurons that 
        has z vaule higher than itself. 

        Tuple ``pos``: 3D postion of the given neuron

        Integer ``i``: index of the given neuron in self.neurons """

        # initialize a list of candidates of connections
        candidates = []

        # start with neurons with higher z values
        start = max(i, self.sizes[0]*self.sizes[1]) # in case of input neurons, start from the hidden layer

        # distance is equavalent to magnitude of difference vector
        for p in self.pos[start:]:
            candidates.append(np.subtract(p, pos))

        # get a list of indices sorted on distance from the given position
        indices = [index for index, value in sorted(enumerate(candidates), key=lambda x:(x[1]**2).sum())]

        # return indices of the first #(self.density) neurons
        return [index + start for index in indices[1:self.density+1]]

    def feedforward(self, imgs):
        """
        Feedforward an array of imgs to the neural network. Given 
        the network is async, feedforward loop might be longer than
        video length """
        
        vid_length = len(imgs)
        self.set_input(imgs[0])
        current = 1

        # feedforward while intermediate signals are not zeros       
        while self.signals.sum():
            # get the non zeros indices
            non_zeros = [i for i, e in enumerate(self.signals) if e!=0]

            # loop through non_zeros
            for i in non_zeros:
                # get outputs from a single neuron
                outputs = self.neurons[i].receive(self.signals[i])
                # clear received signal
                self.signals[i] = 0

                if len(self.neurons[i].outs)!=0: # check if the neuron is at top of the network
                    # store the outputs to signals if the neuron fires
                    if type(outputs)!=int:
                        # store signals to individual neurons that connected to the neuron
                        for j in range(len(self.neurons[i].outs)):
                            self.signals[self.neurons[i].outs[j]] += outputs[j]
                else:
                    # append to output if the neuron is at top of the network
                    self.output.append(outputs)

            # store input signals if available
            if current < vid_length:
                self.set_input(imgs[current])
                current += 1
                
    def set_input(self, img):
        """store a single img to input signals"""
        for i in range(self.sizes[0]):
            for j in range(self.sizes[1]):
                self.signals[i*self.sizes[0]+j] = img[i][j]


class Neuron(object):
    def __init__(self):
        self.nucleus = 0
        self.threshold = .2

    def connect(self, outs):
        self.outs = outs
        if len(outs)!=0:
            self.w = abs(np.random.randn(len(outs)))
        else:
            self.w = 1

    def receive(self, signal):
        # reset previous over-threshold nucleus
        if self.nucleus >= self.threshold:
            self.nucleus = 0

        # fire if nucleus is above threshold
        self.nucleus += signal
        if self.nucleus >= self.threshold:
            return self.fire()
        else:
            return 0

    def normalize(self):
        self.w = self.w/np.linalg.norm(self.w)

    def fire(self):
        self.normalize()
        return self.w * self.nucleus

# if __name__ == '__main__':
# import numpy
# import networks
# net = networks.Network([10,10,10], 30, 100)
# net.feedforward(abs(numpy.random.randn(40, 10, 10)))