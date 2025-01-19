#include "spiker_network.h"
#include <stdlib.h> // calloc
#include "utilities/text_formatter.h"
#include "spiker_network_gpu.cuh"

const float SPIKE_VOLTAGE = 30.0f;
struct Neuron *neurons = NULL;
int neuron_count = 0, total_neuron_spaces = 0;
float step_time = 1.0f;


void init_network(int neuron_count)
{
    // !!! Function does not require best performance (no real-time usage) !!!
    neurons = calloc(neuron_count, sizeof(struct Neuron));
    total_neuron_spaces = neuron_count;
    print_success("Network initialized on CPU!\n");
}

int add_neuron(float v, float a, float b, float c, float d)
{
    // !!! Function does not require best performance (no real-time usage) !!!
    if (neuron_count >= total_neuron_spaces)
    {
        return 1;
    }
    neurons[neuron_count].v = v;
    neurons[neuron_count].a = a;
    neurons[neuron_count].b = b;
    neurons[neuron_count].c = c;
    neurons[neuron_count].d = d;
    neuron_count++;
}

int add_connection(int from, int to, int latency, float weight)
{
    // !!! Function does not require best performance (no real-time usage) !!!
    unsigned int failure = 0, warning = 0;
    if (from >= neuron_count || to >= neuron_count || from < 0 || to < 0)
    {
        print_error("Cannot add neuron connection due to invalid neuron index: ");
        failure = 1;
    }
    if (latency < 0 || latency >= MAX_NEURON_LATENCY)
    {
        print_error("Cannot add neuron connection due to invalid latency: ");
        failure = 1;
    }
    if (weight < -0.5 || weight > 1)
    {
        print_warning("Abnormal weight value for connection: ");
        warning = 1;
    }
    if (failure || warning)
    {
        printf("from: %d, to: %d, latency: %d, weight: %f\n", from, to, latency, weight);
    }
    if (failure)
    {
        return 1;
    }
    size_t i = 0;
    while (i < MAX_NEURON_INPUTS)
    {
        if (neurons[to].inputs[i] == 0)
        {
            neurons[to].inputs[i] = from;
            neurons[to].latencies[i] = latency;
            neurons[to].weights[i] = weight;
            return 0;
        }
        i++;
    }
    print_error("Cannot add neuron connection due to full input list: ");
    printf("from: %d, to: %d, latency: %d, weight: %f\n", from, to, latency, weight);
    return 1;
}

void simulate_step()
{
    // !!! FUNCTION REQUIRES BEST PERFORMANCE (MAY BE REAL-TIME USAGE) !!!
    // TODO: Implement CPU simulation
    
}

void free_network()
{
    // !!! Function does not require best performance (no real-time usage) !!!
    free(neurons);
    neurons = NULL;
    total_neuron_spaces = 0;
}
