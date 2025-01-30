#include "spiker_network.h"
#include <stdlib.h> // calloc
#include "utilities/text_formatter.h"
#include "spiker_network_gpu.h"

const float SPIKE_VOLTAGE = 30.0f;
struct Neuron *neurons = NULL;
int main_neuron_count = 0, main_neuron_spaces = 0, input_neurons = 0, output_neurons = 0;
float step_time = 1.0f;
int allow_self_connections = 0;


int init_network(int input_count, int main_neuron_count, int output_count)
{
    // !!! Function does not require best performance (no real-time usage) !!!
    if (main_neuron_count < output_count)
    {
        print_error("Not enough processing neurons for setting them as outputs. ");
        printf("Asked for: %d processing neurons and %d outputs\n", main_neuron_count, output_count);
        return 1;
    }
    neurons = (struct Neuron *)calloc((size_t)input_count + (size_t)main_neuron_count, sizeof(struct Neuron));
    input_neurons = input_count;
    main_neuron_spaces = main_neuron_count;
    output_neurons = output_count;
    print_success("Network initialized on CPU!\n");
}

int add_neuron(float v, float a, float b, float c, float d)
{
    // !!! Function does not require best performance (no real-time usage) !!!
    if (main_neuron_count >= main_neuron_spaces)
    {
        return 1;
    }
    int current_index = input_neurons + main_neuron_count;
    neurons[current_index].v = v;
    neurons[current_index].a = a;
    neurons[current_index].b = b;
    neurons[current_index].c = c;
    neurons[current_index].d = d;
    main_neuron_count++;
}

int add_connection(int from, int to, int latency, float weight)
{
    // !!! Function does not require best performance (no real-time usage) !!!
    unsigned int failure = 0, warning = 0;
    if (from == to && !allow_self_connections)
    {
        print_error("Cannot add neuron connection to itself: ");
        failure = 1;
    }
    else if (from >= main_neuron_count + input_neurons || to >= main_neuron_count + input_neurons || from < 0 || to < 0)
    {
        print_error("Cannot add neuron connection due to invalid neuron index: ");
        failure = 1;
    }
    else if (latency < 0 || latency >= MAX_NEURON_LATENCY)
    {
        print_error("Cannot add neuron connection due to invalid latency: ");
        failure = 1;
    }
    else if (weight < -0.5 || weight > 1) // TODO: Get these values from property file MIN/MAX
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
    main_neuron_spaces = 0;
}
