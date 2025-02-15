#pragma once

#include <stdio.h> // NULL

#define NEURON_SPIKE_TRAIN_TYPE unsigned long
#define MAX_NEURON_LATENCY sizeof(NEURON_SPIKE_TRAIN_TYPE) * 8
#define EX_IN_RATIO 4.0f

// THESE MUST BE ACCESIBLE ALSO FROM THE GPU
// TODO: transfer to dynamic variables: min_connections, max_connections
#define MIN_NEURON_INPUTS 64
#define MAX_NEURON_INPUTS 64

// Have neurons connected in order. If not then random.
#define SEQUENTIAL_NETWORK 0

struct Neuron
{
    unsigned int incomming_connections;
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
extern "C"
{
#endif

    /// <summary>
    /// Neuron array starts with X input neurons. Continues with N main (process neurons) starting with inhibitory and ending with excitatory. Last Y neurons of array are outputs (should be all excitatory)
    /// [INPUT, INPUT, INPUT, INH_N, EX_N, EX_N, EX_N, EX_N]
    /// </summary>
    extern struct Neuron* neurons;
    extern int first_output_neuron_index;
    extern int main_neuron_count, main_neuron_spaces, input_neurons, output_neurons;
    extern int recommended_excitatory_neuron_count, recommended_inhibitory_neuron_count;
    extern float step_time; // Step time in milliseconds
    extern char* live_spike_array_cpu;
    // Virtual screen dimensions for network 2d size info
    extern int virtual_screen_w, virtual_screen_h;

    // Setting variables
    extern int allow_self_connections;
    extern const float SPIKE_VOLTAGE;
    /// <summary>
    /// Min and max connections neuron can have
    /// </summary>
    extern int min_connections, max_connections;

#ifdef __cplusplus
}
#endif

/// @param main_neuron_count How many neurons can be stored in the network

/// <summary>
/// Initializes the network with enough spaces for the given number of neurons
/// </summary>
/// <param name="input_count">- How many inputs</param>
/// <param name="main_neuron_count">- How many processing neurons</param>
/// <param name="output_count">- How many outputs (must be less than processing neuron count)</param>
int init_network(int input_count, int main_neuron_count, int output_count);

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

/// <summary>
/// Populates neuron network with recommeded ratios and settings for neurons
/// </summary>
/// <returns></returns>
int populate_neuron_network_automatically();


/// <summary>
/// Automatically connects neurons within neural network
/// </summary>
/// <returns></returns>
int connect_neuron_network_automatically();

/// <summary>
/// Can test compute performance of the network
/// </summary>
/// <param name="iterations">How many times to do check to get average</param>
/// <returns>Step time in milliseconds</returns>
float get_step_performance(unsigned int iterations);

/// @brief Simulates a single step of the network
void simulate_step();

/// @brief Starts the simulation in background
void start_simulation();

/// @brief Stops the simulation
void stop_simulation();

/// @brief Frees the memory allocated for the network
void free_network();