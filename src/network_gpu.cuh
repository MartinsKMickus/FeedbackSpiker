#pragma once

#include <cuda_runtime.h> // For CUDA runtime API (__device__, __global__, etc.)
#include "network_cpu.h" // For Neuron struct

/// @brief Simulates a single step of the network
/// @param neurons Neuron array
/// @param step_time Step time to simulate in milliseconds
__global__ void update_neuron(struct Neuron *neurons, float step_time);