#pragma once


// #include <cuda.h>
#include "spiker_network.h" // For Neuron struct
#include <cuda_runtime.h> // For CUDA runtime API (__device__, __global__, etc.)

extern unsigned int GPU_READY;
// TODO: Add flag to check for new changes in the network on CPU side

//extern struct Neuron *gpu_neurons;

/// @brief Simulates a single step of the network
/// @param neurons Neuron array
/// @param step_time Step time to simulate in milliseconds
//__global__ void update_neuron(Neuron* neurons, unsigned int max_index, unsigned int min_index);


// This code will be called from C code. Need to do this because of CUDA being in C++
#ifdef __cplusplus
extern "C" 
{
#endif

	/// @brief Prepares the GPU for the network simulation
	/// @return 0 on success, 1 on failure
	int init_gpu_network();

	int simulate_gpu_step();

	/// <summary>
	/// Copies CPU input neuron data onto GPU
	/// </summary>
	/// <returns></returns>
	int refresh_gpu_inputs_from_cpu();

	int transfer_gpu_spike_array_to_cpu();

	void free_gpu_network();

#ifdef __cplusplus
}
#endif
