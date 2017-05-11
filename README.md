# Sleepbot
### Yuzhong Huang, Dhash Shrivathsa
# Project Documentation
## Abstract
Sleepbot is our project for improvements in generalization error for convolutional neural networks. Inspired by neuroscience, we aimed to build heuristics to speed up training or improve test set performance. We built visualization, logging, graphing, and statistics for a CNN to this effect. The structure of our code allows training processes to be transferred to some extent, through training a secondary network around the training process, though we did not thouroughly investigate this.

## Background
Initially inspired by neuroscience, deep learning have achieved great results in various research area including Computer Vision, Natural Language Processing and Robot Controls. Driven by the demand for high speed performance, most deep learning frameworks are confined to a layer-based form, which is efficient and compatible with the current computing system. There is merit to changing this architecture, and one way is to take a critical look at backpropogation.

### The training process
Neural networks are trained by

1. Feedforward a batch of images (a tunable "chunk" of the dataset)
2. Backpropogate the errors of the collective batch
3. Repeat for n epochs

In the middle is the most important step, what is considered "training". Backpropogation calculates the error magnitude for each neuron and updates it's weights in closed form. Our method intercepts weights in between training batches, so between each step of backpropogation.

### Backpropogation
What is wrong with backpropogation? It is a formative algorithim, used to propogate weight update throught the network. Unfortunatley, its biggest strength is also its biggest weakness. A closed form iterateve solution cannot observe time-series steps, and tends to overfit or maintain the current trend

### Proposed solution
Looking at the time-series structure for how the weights of neurons evolve seems like a good proxy for speeding up training. Taking this further can mean training a LSTM-RNN on the layer weights for an approach. In this work, we only propose simple numerical heuristics as to how to speed training and improve generalization error.

## Visualization of performance

### Visualization of the training process
Given the inherent structure to convolutional neural networks, there exist several ways to visualize what the process looks like. We chose two, the image formed as the neural net is shown all 1's (the magnitude of the contribution of each neuron), and the norm of their weights through time.

### Trend in neuron
As we view the neurons training, we can see that each has a predetermined distribution across which this is spread. The general widening trend with little reorgainization implies training is done and the network has reached an equilibrium state, with only miniscule incremental gains in performance. However, early identification of the asymptotic limit is a possible heuristic that we can use to speed up training. Another possible metric is the delta in relative spread across the neuron norms. This manifests in the graph as the ratio of the norms staying roughly equal as the network saturates.

## Performance evaluation of heuristics

### Baseline performance and success measures
we can see what a standard no-sleep run looks like, as well as one that has been corrupted by noise. The down-and-right nature of the random noise curve is indicitave of worse performance, and the "fuzz" along the trained network is random noise being injected, and the network compensating with another step of backpropogation. Geometrically speaking, it is easy to see what success looks like, relative up-and-left shift of the line.

The spike at the beginning deserves mention, as it's a lucky configureation found by chance. We can see it quickly drop off as the opposite happens, a very unlucky sleep operation.
  
Ideally, we're able to replicate the early boost given by a random sleep (in this case).

## Results
We started out with an intent to simulate and implement a “sleep” phase for deep learning, during which the model takes no external data and change its weight through some signals in the system. As we researched on sleep, we found that sleep is a very complex behavior that influences mainly the chemical transmission between neuron. It is very hard to generalize and simulate a sleep phase with a single signal neural network system.   

## Future work
We still have to try transfer learning with a LSTM-RNN, and more heuristsics are always good to try. Now that we have a toolkit, trying out new things is easy, so we should iterate and try more heuristics. Sleep in a neural network is a challenging thing.    

## Project Story 1

During the first phase of our project, we aimed to build a tool set to visualize the network's parameters and a system to assess our algorithm. We initially tried to write our own implementation of deep learning framework, but it seems to be very messy therefore we used Keras and write some utility functions on top. As for the visualization part, we initially tried to visualize the network in a three dimensional model and color the connections between neurons, but there are two problems associated with this method: First, the visualization looks very messy and we cannot really tell which weights is associated with which connection. Second, we cannot fit a multi-channel convolutional neural network into three dimension model since each channel is an independent dimension. Therefore, we define a recursive weight of a neuron by feedforwarding a matrix with all 1 and compile it into a video. However, the weights video seems random and cannot be used for gaining intuition.

## Project Story 2

During the second phase of our project, we want to experiments some simple methods, added between each batch backpropagation to see if the algorithm improves the generalization error of the network. First, we tried to add random noises to the network. It didn't work very well just as we thought. But we can clearly see the difference in ways of how neurons converge: In a normal backpropagation process, weights of a network oscillates around the final optimal value with exponentially decreasing amplitude. However, by adding proportional noise to the weights, although the weights finally reached a similar steady state, the route it took to reach that is very different. Another simple method we tried is to square the weights and normalize it for each neuron in the network. This method actually did a better job in reducing generalization error. On the other hand, we realized that in order to try some idea inspired by neuron science fact, we need flexibility for how neurons connects with each other that Keras does not provide. Therefore, we continued to write our own deep learning framework, in which neurons are the basic units, not layers. Since it is not layer-based, we cannot use palatalization that we gained from layer-based deep learning framework, it is a lot harder to implement.    
