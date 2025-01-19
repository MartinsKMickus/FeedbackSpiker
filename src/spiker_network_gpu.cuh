#pragma once

#include <cuda_runtime.h> // For CUDA runtime API (__device__, __global__, etc.)
// #include <cuda.h>
#include "spiker_network.h" // For Neuron struct

extern unsigned int GPU_READY;
// TODO: Add flag to check for new changes in the network on CPU side

extern struct Neuron *gpu_neurons;

/// @brief Simulates a single step of the network
/// @param neurons Neuron array
/// @param step_time Step time to simulate in milliseconds
__global__ void update_neuron(struct Neuron *neurons, float step_time);

// This code will be called from C code. Need to do this because of CUDA being in C++
#ifdef __cplusplus
extern "C" {
#endif

/// @brief Prepares the GPU for the network simulation
/// @return 0 on success, 1 on failure
int init_gpu_network();

int simulate_gpu_step();

void free_gpu_network();

#ifdef __cplusplus
}
#endif

