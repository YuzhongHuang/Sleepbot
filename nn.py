import pycuda.autoinit
import pycuda.gpuarray as gpuarray
import pycuda.driver as drv
import numpy as np

import skcuda.linalg as culinalg
import skcuda.misc as cumisc

import typing
from copy import deepcopy, copy
from attr import attrs, attrib


import skcuda.linalg as linalg
linalg.init()
import numpy as np

Array = np.array

def sigmoid(x,deriv=False):
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

@attrs
class Neuron:
    params = attrib() 
    h = []
        
    def update(self, new_p: Array):
        self.h.append(copy(self.params))
        self.params = new_p
        
    def run(self, X: Array) -> Array:
        X_g = gpuarray.to_gpu(X)
        params_g = gpuarray.to_gpu(self.params)
        return self.activation(linalg.dot(X_g, params_g).get())
    
    @staticmethod
    def activation(X):
        pass
    

class Sigmoid_Neuron(Neuron):
    @staticmethod
    def activation(X):
        return sigmoid(X)
        
