"""

Given the hierachy of neural network's recursive weights,
generate a series of weight visualization video for each layer

Yuzhong Huang
"""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from pylab import *

import pickel

# fake layer data of 32 by 32 and 100 time series
time = 100
layer = np.random.randn(100,1,32)

def write_layer_animation(layer, x, y, t, name, dpi=100, fps=5):
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.set_aspect('equal')
    ax.get_xaxis().set_visible(False)
    ax.get_yaxis().set_visible(False)

    im = ax.imshow(np.zeros(shape=(x,y)),cmap='gray',interpolation='nearest')
    im.set_clim([0,1])
    fig.set_size_inches([5,5])

    plt.tight_layout()

    def update_img(n):
        im.set_data(layer[n])
        return im

    ani = animation.FuncAnimation(fig,update_img,t,interval=t)
    writer = animation.writers['ffmpeg'](fps=fps)

    ani.save(name+'.mp4',writer=writer,dpi=dpi)


write_layer_animation(layer, 1, 32, 100, "demo")
