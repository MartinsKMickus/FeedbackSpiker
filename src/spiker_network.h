#pragma once

#include <stdio.h> // NULL

#define NEURON_SPIKE_TRAIN_TYPE unsigned long
#define MAX_NEURON_LATENCY sizeof(NEURON_SPIKE_TRAIN_TYPE) * 8

// THESE MUST BE ACCESIBLE ALSO FROM THE GPU
#define MAX_NEURON_INPUTS 1024

struct Neuron
{
    unsigned int inputs[MAX_NEURON_INPUTS]; // Other neuron indexes that are connected to this one
    unsigned int latencies[MAX_NEURON_INPUTS]; // Latencies of connections
    float weights[MAX_NEURON_INPUTS]; // Weights of connections
    // Izhikevich model parameters
    float v; // Membrane potential
    float u; // Recovery variable
    float a, b, c, d; // Parameters
    // I will be calculated later
    // TODO: Make diagnostics to check struct size
    NEURON_SPIKE_TRAIN_TYPE spike_train; // Train of spikes in bitwise (can representg 64 steps)
};

// Using extern to avoid multiple definition errors and for sharing the variables between files (global variables)
// If static is usaed then each file will have its own copy of the variable
// Checking if compiled as cpp because for some symbols we need to specify linking rules
#ifdef __cplusplus
extern "C" {
#endif

extern const float SPIKE_VOLTAGE;
extern struct Neuron *neurons;
extern int neuron_count, total_neuron_spaces;
extern float step_time; // Step time in milliseconds

#ifdef __cplusplus
}
#endif

/// @brief Initializes the network with the given number of neurons
/// @param neuron_count How many neurons can be stored in the network
void init_network(int neuron_count);

/// @brief Adds a neuron to the network. Follow cell type distribution of these values. For input neurons set all values to 0 as they won't be used.
/// @param v initial membrane potential
/// @param a recovery rate (excitatory ~0.02, inhibitory ~0.02-0.1)
/// @param b sensitivity of the recovery variable to v changes (excitatory ~0.2, inhibitory ~0.2-0.25)
/// @param c membrane potential reset value (around -65mV)
/// @param d After-spike reset of the recovery variable (excitatory 8, inhibitory 2)
/// @return 0 if neuron is added, 1 adding failed
int add_neuron(float v, float a, float b, float c, float d);

/// @brief Adds a connection between two neurons
/// @param from Index of the neuron that sends the signal
/// @param to Index of the neuron that receives the signal
/// @param latency Latency of the connection in steps before the signal reaches the target
/// @param weight Weight of the connection (negative for inhibitory, positive for excitatory neurons) -0.5 <= weight <= 1
/// @return 0 if connection is added, 1 adding failed
int add_connection(int from, int to, int latency, float weight);
// TODO: Defife input and output neurons?

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