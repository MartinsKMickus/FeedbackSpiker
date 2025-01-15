#pragma once
#include <cuda_runtime.h> // For CUDA runtime API (__device__, __global__, etc.)

const float SPIKE_VOLTAGE = 30.0f;

struct Neuron
{
    unsigned int inputs[1024]; // Other neuron indexes that are connected to this one
    float weights[1024]; // Weights of connections
    // Izhikevich model parameters
    float v; // Membrane potential
    float u; // Recovery variable
    float a, b, c, d; // Parameters
    // I will be calculated later
    // TODO: Make diagnostics to check struct size
    unsigned long spike_train; // Train of spikes in bitwise (can representg 64 steps)
};

struct Neuron *neurons; // Neurons represented as an array
int neurons_count;

// Connections represented as a matrix
int **connections;

void add_neuron(int index, float v, float a, float b, float c, float d);
void add_connection(int from, int to, int latency, float weight);
// TODO: Defife input and output neurons?

__device__ int update_neuron(struct Neuron *neurons);

float get_step_performance();

void simulate_step();
void start_simulation();
void stop_simulation();


void free_network();