# Binary Networks

This repository contains code for building toy neural networks with some 
nonstandard approximations:
* The neurons have binary outputs of ±1, but real valued weights and 
  thresholds
* The 'activation function' is a step function at the threshold
* Derivatives of the activation function are (usually) ignored in 
  backpropagation

The structure of the underlying forward- and backpropagation algorithms do not
make assumptions about network topology, and should allow connections across 
layers. No attempt is currently made to resolve topologies that form loops, but
this will be explored.

Otherwise, this implements a standard backpropagation algorithm to train a 
network of neurons, and examples for training and visualizing such networks.
