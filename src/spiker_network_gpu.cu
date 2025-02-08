#include "spiker_network_gpu.h"
#include "spiker_network.h"
#include "utilities/text_formatter.h"
#include "device_launch_parameters.h"
#include <stdlib.h> //exit

// nvcc does not seem to like variadic macros, so we have to define
// one for each kernel parameter list:
// TL;DR: This is for IDE to supress formatting errors
#ifdef __CUDACC__
#define KERNEL_ARGS2(grid, block) <<< grid, block >>>
#define KERNEL_ARGS3(grid, block, sh_mem) <<< grid, block, sh_mem >>>
#define KERNEL_ARGS4(grid, block, sh_mem, stream) <<< grid, block, sh_mem, stream >>>
#else
#define KERNEL_ARGS2(grid, block)
#define KERNEL_ARGS3(grid, block, sh_mem)
#define KERNEL_ARGS4(grid, block, sh_mem, stream)
#endif

unsigned int GPU_READY = 0;
Neuron *gpu_neurons;
char* live_spike_array_gpu;
__device__ float SPIKE_VOLTAGE_GPU;
__device__ int MAX_NEURON_INPUTS_GPU;
__device__ float step_time_gpu = 1.0f;

__global__ void update_neuron(Neuron *neurons, unsigned int max_index, unsigned int min_index)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= max_index || index < min_index)
    {
        return;
    }
    float I = 0;
    int inputIdx, latencyBit;
    for (int i = 0; i < neurons[index].incomming_connections; i++)
    {
        inputIdx = neurons[index].inputs[i] - 1; // 0-based index afterwards
        latencyBit = 1 << neurons[index].latencies[i];
        // If there is spike with specific latency
        if (neurons[inputIdx].spike_train & latencyBit) {
            I += neurons[index].weights[i]; // TODO: FIXME remove value
            if (neurons[index].weights[i] < 0.00001 && neurons[index].weights[i] > 0)
            {
                printf("Neuron connection dying. Detected weight: %f\n", neurons[index].weights[i]);
            }
            if (neurons[index].weights[i] > -0.00001 && neurons[index].weights[i] < 0)
            {
                printf("Neuron connection dying. Detected weight: %f\n", neurons[index].weights[i]);
            }
            if (neurons[index].weights[i] < -0.9999)
            {
                printf("Neuron poisonous. Detected weight: %f\n", neurons[index].weights[i]);
            }
        }

    }
    //v = v + time_delta * (0.04 * v^2 + 5 * v + 140 - u + I)
    // TODO: Validate mathematics (code review)!!!
    /*float core_calculation = 0.04f * neurons[index].v * neurons[index].v + 5.0f * neurons[index].v + 140.0f - neurons[index].u + I;
    core_calculation *= step_time_gpu;
    neurons[index].v += core_calculation;*/
    // TODO: fix hardcoded time
    neurons[index].v = neurons[index].v + step_time_gpu * (0.04f * neurons[index].v * neurons[index].v + 5.0f * neurons[index].v + 140.0f - neurons[index].u + I);
    // u = u + time_delta * (a * (b * v - u));
    // TODO: Validate mathematics (code review)!!!
    neurons[index].u = neurons[index].u + (step_time_gpu * (neurons[index].a * (neurons[index].b * neurons[index].v - neurons[index].u)));
    //printf("Voltrage %f \n", neurons[index].v);
    if (neurons[index].v >= SPIKE_VOLTAGE_GPU)
    {
        // LEARNING!!!!!!!!!!!!!!!!!
        for (int i = 0; i < neurons[index].incomming_connections; i++)
        {
            inputIdx = neurons[index].inputs[i] - 1;
            latencyBit = 1 << neurons[index].latencies[i];
            // If there is spike with specific latency
            if (neurons[index].weights[i] > 0)
            {
                if (neurons[inputIdx].spike_train & latencyBit)
                {
                    neurons[index].weights[i] += 0.00025f;
                }
                else if (neurons[inputIdx].spike_train & latencyBit << 1)
                {
                    neurons[index].weights[i] += 0.00015f;
                }
                else if (neurons[inputIdx].spike_train & latencyBit << 2)
                {
                    neurons[index].weights[i] += 0.00005f;
                }
                else
                {
                    neurons[index].weights[i] *= 0.99999f;
                }
            }
            else
            {
                if (neurons[inputIdx].spike_train & latencyBit)
                {
                    neurons[index].weights[i] *= 0.99995f;
                }
                else if (neurons[inputIdx].spike_train & latencyBit << 1)
                {
                    neurons[index].weights[i] *= 0.999925f;
                }
                else if (neurons[inputIdx].spike_train & latencyBit << 2)
                {
                    neurons[index].weights[i] *= 0.999991f;
                }
            }

        }
        neurons[index].v = neurons[index].c;
        neurons[index].u = neurons[index].u + neurons[index].d;
        neurons[index].spike_train |= 1 << 0; // Set the first bit to 1 because the neuron spiked
    }
    else
    {
        // LEARNING NEGATIVE!!!!!!!!!!!!!!!!!
        for (int i = 0; i < neurons[index].incomming_connections; i++)
        {
            inputIdx = neurons[index].inputs[i] - 1;
            latencyBit = 1 << neurons[index].latencies[i];
            // If there is spike with specific latency
            if (neurons[index].weights[i] < 0)
            {
                if (neurons[inputIdx].spike_train & latencyBit)
                {
                    neurons[index].weights[i] -= 0.00005f;
                }
                else if (neurons[inputIdx].spike_train & latencyBit << 1)
                {
                    neurons[index].weights[i] -= 0.000025f;
                }
                else if (neurons[inputIdx].spike_train & latencyBit << 2)
                {
                    neurons[index].weights[i] -= 0.00001f;
                }
                else
                {
                    neurons[index].weights[i] *= 0.99999f;
                }
            }
            else
            {
                if (neurons[inputIdx].spike_train & latencyBit)
                {
                    neurons[index].weights[i] *= 0.99995f;
                }
                else if (neurons[inputIdx].spike_train & latencyBit << 1)
                {
                    neurons[index].weights[i] *= 0.99998f;
                }
                else if (neurons[inputIdx].spike_train & latencyBit << 2)
                {
                    neurons[index].weights[i] *= 0.99999f;
                }
            }

        }
    }
    return;
}

__global__ void move_to_next_step(Neuron* neurons, unsigned int max_index, unsigned int min_index)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= max_index)
    {
        return;
    }
    neurons[index].spike_train <<= 1;
}

__global__ void gather_spike_info(Neuron* neurons, char* spike_array, unsigned int input_count, unsigned int max_index)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index >= max_index)
    {
        return;
    }
    if (neurons[index].spike_train & 1 << 1)
    {
        if (index < input_count)
        {
            spike_array[index] = 128;
        }
        else
        {
            spike_array[index] = 0xFF;
        }
    }
    else
    {
        spike_array[index] = 0;
    }
    //spike_array[index] = 0xFF;
    return;
}

extern "C" int check_cuda_failure(cudaError_t err, const char * message)
{
    if (err != cudaSuccess)
    {
        print_error("CUDA failure ");
        printf("(%s) Message: %s\n",message, cudaGetErrorString(err));
        exit(1);
    }
}

extern "C" int init_gpu_network()
{
    cudaMemcpyToSymbol((const void*)&MAX_NEURON_INPUTS_GPU, &max_connections, sizeof(max_connections));
    //cudaMemcpy(&MAX_NEURON_INPUTS_GPU, &max_connections, sizeof(max_connections), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    check_cuda_failure(cudaGetLastError(), "MAX_INPUTS");
    cudaMemcpyToSymbol((const void*)&SPIKE_VOLTAGE_GPU, &SPIKE_VOLTAGE, sizeof(SPIKE_VOLTAGE));
    cudaDeviceSynchronize();
    check_cuda_failure(cudaGetLastError(), "SPIKE_VOLGATE");
    cudaMalloc(&gpu_neurons, sizeof(Neuron) * (main_neuron_count + input_neurons));
    cudaDeviceSynchronize();
    check_cuda_failure(cudaGetLastError(), "Malloc GPU neurons");
    cudaMalloc(&live_spike_array_gpu, sizeof(char) * (main_neuron_count + input_neurons));
    cudaDeviceSynchronize();
    check_cuda_failure(cudaGetLastError(), "Malloc GPU spike array");
    cudaMemcpy(gpu_neurons, neurons, sizeof(Neuron) * (main_neuron_count + input_neurons), cudaMemcpyHostToDevice);
    cudaDeviceSynchronize();
    check_cuda_failure(cudaGetLastError(), "Copy GPU neurons");

    
    GPU_READY = 1;
    print_success("GPU network initialized\n");
    return 0;
}

extern "C" int simulate_gpu_step()
{
    if (!GPU_READY)
    {
        return 1;
    }
    // TODO: Implement atomic block or mutex of some sort for other processes to probe neurons and insert data
    move_to_next_step KERNEL_ARGS2((input_neurons + main_neuron_count) / GPU_BLOCKS + 1, GPU_BLOCKS)(gpu_neurons, input_neurons + main_neuron_count, input_neurons);
    cudaDeviceSynchronize();
    check_cuda_failure(cudaGetLastError(), "Shifting spikes");
    update_neuron KERNEL_ARGS2((input_neurons + main_neuron_count) / GPU_BLOCKS + 1, GPU_BLOCKS)(gpu_neurons, input_neurons + main_neuron_count, input_neurons);
    cudaDeviceSynchronize();
    check_cuda_failure(cudaGetLastError(), "Neuron update");
    // TODO: Optimize (do not copy each time)
    //cudaMemcpy(neurons, gpu_neurons, sizeof(Neuron) * main_neuron_count, cudaMemcpyDeviceToHost);
    return 0;
}

int refresh_gpu_inputs_from_cpu()
{
    // TODO: Implement locking mechanism
    cudaMemcpy(gpu_neurons, neurons, sizeof(Neuron) * input_neurons, cudaMemcpyHostToDevice);
    return 0;
}

int transfer_gpu_spike_array_to_cpu()
{
    if (!GPU_READY)
    {
        return 1;
    }
    gather_spike_info KERNEL_ARGS2((input_neurons + main_neuron_count) / GPU_BLOCKS + 1, GPU_BLOCKS)(gpu_neurons, live_spike_array_gpu, input_neurons, input_neurons + main_neuron_count);
    cudaDeviceSynchronize(); 
    cudaMemcpy(live_spike_array_cpu, live_spike_array_gpu, sizeof(char) * (input_neurons + main_neuron_count), cudaMemcpyDeviceToHost);
    return 0;
}

extern "C" void free_gpu_network()
{
    cudaFree(gpu_neurons);
    cudaFree(live_spike_array_gpu);
    GPU_READY = 0;
    print_success("GPU network freed\n");
    return;
}
