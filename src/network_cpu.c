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
