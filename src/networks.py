import numpy as np

class Network(object):
    def __init__(self, sizes, density, num):
        self.sizes = sizes
        self.density = density
        self.output = []
        self.neurons = [Neuron() for i in range(sizes[0]*sizes[1] + num)] # (input neurons + num) of neurons

        # initialize neuron's positions and form connections of neurons
        self.pos = []
        self.init_pos()
        self.form_connection()

        # initialize signals that store all the intermediate inputs to individual neurons
        self.signals = np.zeros(len(neurons))

    def init_pos(self):
        for i in self.sizes[0]:
            for j in self.sizes[1]:
                self.pos.append((i,j,0))

        xs = np.random.uniform(low=0, high=sizes[0], size=(num,))
        ys = np.random.uniform(low=0, high=sizes[1], size=(num,))
        zs = np.random.uniform(low=0, high=sizes[2], size=(num,))
        self.pos += ([(x,y,z) for x, y, z in zip(xs, ys, zs)])

    def form_connection(self):
        for i in range(len(self.pos)):
            self.neurons[i].connect(find_connection(self.pos(i)))

    def magnitude(pos):
        return (pos**2).sum()

    def find_connection(self, pos):
        l = []
        for p in self.pos:
            l.append(numpy.subtract(p, pos))
        sorted(l, key=magnitude)[:self.density]

    def feedforward(self, img):
        # store input signals
        for i in self.sizes[0]:
            for j in self.sizes[1]:
                self.signals[i*self.sizes[0]+j] = img[i][j]

        # feedforward while signals are not zeros       
        while self.signals.sum():
            # get the non zeros indices
            non_zeros = [i for i, e in enumerate(self.signals) if e!=0]

            # loop through non_zeros
            for i in non_zeros:
                # get outputs from a single neuron
                outputs = self.neurons[i].receive(self.signals[i])

                if self.neurons[i].outs:
                    # store the outputs to signals if the neuron fires
                    if outputs:
                        # store signals to individual neurons that connected to the neuron
                        for j in range(len(self.neurons[i].outs)):
                            self.signals[self.neurons[i].outs(j)] = outputs[j]
                else:
                    self.output.append(outputs)


class Neuron(object):
    def __init__(self):
        self.nucleus = 0
        self.threshold = 1

    def connect(self, outs):
        self.outs = outs
        self.w = np.random.randn(len(outs))

    def receive(self, signal):
        # reset previous over-threshold nucleus
        if self.nucleus > self.threshold:
            self.nucleus = 0

        # fire if nucleus is above threshold
        self.nucleus += signal
        if self.nucleus > self.threshold:
            self.fire()
        else:
            return 0

    def normalize(self):
        self.w = self.w/np.linalg.norm(self.w)

    def fire(self):
        self.normalize()
        return self.w * self.nucleus