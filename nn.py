#import pycuda.autoinit
#import pycuda.gpuarray as gpuarray
#import pycuda.driver as drv
#import skcuda.linalg as culinalg
#import skcuda.misc as cumisc

import typing
from copy import deepcopy, copy
from attr import attrs, attrib


#import skcuda.linalg as linalg
#linalg.init()
import numpy as np

Array = np.array
Float = np.float32

@attrs
class Intrinsics:
    default_thresh = attrib()
    thresh = default_thresh

def sigmoid(x: Float, deriv=False) -> Float:
    if(deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

@attrs
class Neuron:
    params = attrib() 
    intrinsics = attrib()
    h = []
        
    def update(self, new_p: Array):
        self.h.append(copy(self.params))
        self.params = new_p
        
    def run(self, X: Array) -> Float:
        return self.activation(X.sum(), self.intrinsics)
    
    def activation(X):
        pass
    

class Sigmoid_Neuron(Neuron):
    @staticmethod
    def activation(X: Array) -> Float:
        return sigmoid(X)

class RELU_Neuron(Neuron):
    def activation(self, X: Array) -> Float:
        sum_X = X.sum()
        if self.intrinsics.thresh < sum_X:
            old_thresh = self.intrinsics.thresh
            self.intrinsics.reset_thresh()
            return self.intrinsic.thresh - sum_X
        else:
            self.inrinsics.thresh -= sum_X
            return 0
               
