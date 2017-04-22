import keras
from keras.applications import VGG19, InceptionV3
from keras.preprocessing.image import img_to_array
from keras.preprocessing.image import load_img
import numpy as np

from keras.layers.convolutional import Conv2D
from keras.engine.topology import InputLayer
from keras.layers.pooling import MaxPooling2D

from attr import attrs, attrib


@attrs
class GraphableNeruon(object):
    weights = attrib()
    bias = attrib()
    layer = attrib()
    neuron_number = attrib()
    layer_type = attrib()


def get_neurons(net):
    neurons = []
    for i, layer in enumerate(net.layers):
        if type(layer) is Conv2D:
            wts, bias = layer.get_weights()
            for nn, (w, b) in enumerate(zip(wts.T, bias)):
                neurons.append(GraphableNeruon(w, b, i, nn, layer.get_config()))
    return neurons


def set_neurons(neurons, net):
    last_neuron = neurons[0]
    layer = net.layers[last_neuron.layer]
    layer_weights, layer_bias = layer.get_weights()
    layer_weights = layer_weights.T
    for n in neurons:
        if not last_neuron.layer == n.layer:
            layer.set_weights([layer_weights.T, layer_bias])
            layer = net.layers[n.layer]
            layer_weights, layer_bias = layer.get_weights()
            layer_weights = layer_weights.T
            last_neuron = n
        
        layer_weights[n.neuron_number] = n.weights
        layer_bias[n.neuron_number] = n.bias
        