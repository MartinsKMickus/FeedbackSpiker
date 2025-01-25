#include "spiker_network_gpu.h"
#include "spiker_network.h"
#include "utilities/text_formatter.h"

unsigned int GPU_READY = 0;
Neuron *gpu_neurons;
__device__ float SPIKE_VOLTAGE_GPU;
__device__ int MAX_NEURON_INPUTS_GPU;

__global__ void update_neuron(Neuron *neurons, float step_time)
{
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    float I = 0;
    size_t i = 0;
    while (neurons[index].inputs[i] != 0 && i < MAX_NEURON_INPUTS_GPU)
    {
        I += (neurons[neurons[index].inputs[i]].spike_train & (1 << neurons[index].latencies[i])) * neurons[index].weights[i];
        i++;
    }
    // v = v + time_delta * (0.04 * v^2 + 5 * v + 140 - u + I)
    neurons[index].v = neurons[index].v + step_time * (0.04 * neurons[index].v * neurons[index].v + 5 * neurons[index].v + 140 - neurons[index].u + I);
    // u = u + time_delta * (a * (b * v - u));
    neurons[index].u = neurons[index].u + step_time * (neurons[index].a * (neurons[index].b * neurons[index].v - neurons[index].u));
    if (neurons[index].v >= SPIKE_VOLTAGE_GPU)
    {
        neurons[index].v = neurons[index].c;
        neurons[index].u = neurons[index].u + neurons[index].d;
        neurons[index].spike_train |= 1 << 0; // Set the first bit to 1 because the neuron spiked
    }
    return;
}

extern "C" int init_gpu_network()
{
    cudaMemcpyFromSymbol(&MAX_NEURON_INPUTS_GPU, MAX_NEURON_INPUTS, sizeof(MAX_NEURON_INPUTS));
    cudaMemcpyFromSymbol(&SPIKE_VOLTAGE_GPU, SPIKE_VOLTAGE, sizeof(SPIKE_VOLTAGE));
    cudaMalloc(&gpu_neurons, sizeof(Neuron) * neuron_count);
    cudaMemcpy(gpu_neurons, neurons, sizeof(Neuron) * neuron_count, cudaMemcpyHostToDevice);

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
    update_neuron<<<neuron_count / 256 + 1, 256>>>(gpu_neurons, 0.1);
    // cudaDeviceSynchronize();
    cudaMemcpy(neurons, gpu_neurons, sizeof(Neuron) * neuron_count, cudaMemcpyDeviceToHost);
    return 0;
}

extern "C" void free_gpu_network()
{
    cudaFree(gpu_neurons);
    GPU_READY = 0;
    print_success("GPU network freed\n");
    return;
}
