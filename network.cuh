#pragma once
#include <cuda_runtime.h> // For CUDA runtime API (__device__, __global__, etc.)

const float SPIKE_VOLTAGE = 30.0f;

struct Neuron
{
    unsigned int inputs[1024]; // Other neuron indexes that are connected to this one
    unsigned int latencies[1024]; // Latencies of connections
    float weights[1024]; // Weights of connections
    // Izhikevich model parameters
    float v; // Membrane potential
    float u; // Recovery variable
    float a, b, c, d; // Parameters
    // I will be calculated later
    // TODO: Make diagnostics to check struct size
    unsigned long spike_train; // Train of spikes in bitwise (can representg 64 steps)
};

struct Neuron *neurons = NULL; // Neurons represented as an array
int neuron_count = 0, initialized_neuron_spaces = 0;
float step_time = 1.0f; // Step time in milliseconds

/// @brief Initializes the network with the given number of neurons
/// @param neuron_count How many neurons can be stored in the network
void init_network(int neuron_count);

/// @brief Adds a neuron to the network. Follow cell type distribution of these values
/// @param index Add the neuron to the given index of all neuron list
/// @param v initial membrane potential
/// @param a recovery rate (excitatory ~0.02, inhibitory ~0.02-0.1)
/// @param b sensitivity of the recovery variable to v changes (excitatory ~0.2, inhibitory ~0.2-0.25)
/// @param c membrane potential reset value (around -65mV)
/// @param d After-spike reset of the recovery variable (excitatory 8, inhibitory 2)
void add_neuron(int index, float v, float a, float b, float c, float d);

/// @brief Adds a connection between two neurons
/// @param from Index of the neuron that sends the signal
/// @param to Index of the neuron that receives the signal
/// @param latency Latency of the connection in steps before the signal reaches the target
/// @param weight Weight of the connection (negative for inhibitory, positive for excitatory neurons) -0.5 <= weight <= 1
void add_connection(int from, int to, int latency, float weight);
// TODO: Defife input and output neurons?

/// @brief Simulates a single step of the network
/// @param neurons Neuron array
/// @param step_time Step time to simulate in milliseconds
__device__ void update_neuron(struct Neuron *neurons, float step_time);

/// @brief Can test compute performance of the network
/// @return Step time in milliseconds
float get_step_performance();

/// @brief Simulates a single step of the network
void simulate_step();

/// @brief Starts the simulation in background
void start_simulation();

/// @brief Stops the simulation
void stop_simulation();

/// @brief Frees the memory allocated for the network
void free_network();