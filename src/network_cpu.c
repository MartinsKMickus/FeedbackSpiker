#include "network_cpu.h"
#include <stdlib.h> // calloc


const float SPIKE_VOLTAGE = 30.0f;
struct Neuron *neurons = NULL;
int neuron_count = 0, total_neuron_spaces = 0;
float step_time = 1.0f;


void init_network(int neuron_count)
{
    neurons = calloc(neuron_count, sizeof(struct Neuron));
    total_neuron_spaces = neuron_count;
}

int add_neuron(float v, float a, float b, float c, float d)
{
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

void free_network()
{
    free(neurons);
    neurons = NULL;
    total_neuron_spaces = 0;
}
