#pragma once

struct Neuron
{
    // Izhikevich model parameters
    float v; // Membrane potential
    float u; // Recovery variable
    float a, b, c, d; // Parameters
    // I will be calculated later
};

struct Neuron *neurons; // Neurons represented as an array
int neurons_count;

// Connections represented as a matrix
int **connections;

void add_neuron(int index, float v, float a, float b, float c, float d);
void add_connection(int from, int to, int latency, float weight);
// TODO: Defife input and output neurons?

__device__ void update_neuron(Neuron *neurons);

float get_step_performance();

void simulate_step();
void start_simulation();
void stop_simulation();


void free_network();