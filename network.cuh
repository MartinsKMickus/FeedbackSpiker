#pragma once

struct Neuron
{
    // Izhikevich model parameters
    float v; // Membrane potential
    float u; // Recovery variable
    float a, b, c, d; // Parameters
    // Excitatory and inhibitory connections?
};

struct Neuron *neurons; // Neurons represented as an array

// Connections represented as a matrix
int **connections;