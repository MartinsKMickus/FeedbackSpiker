#include "spiker_network.h"
#include <stdlib.h> // calloc
#include "utilities/text_formatter.h"
#include "spiker_network_gpu.h"
#include "neuron_properties.h"
#include "utilities/code_measurements.h"
#include "math.h"

struct Neuron* neurons = NULL;
int first_output_neuron_index = 0;

/// <summary>
/// MUST USE variables
/// </summary>
int main_neuron_count = 0, main_neuron_spaces = 0, input_neurons = 0, output_neurons = 0;

/// <summary>
/// INFORMATIVE variables
/// </summary>
int recommended_excitatory_neuron_count = 0, recommended_inhibitory_neuron_count = 0;
float step_time = 1.0f;
char* live_spike_array_cpu;
int virtual_screen_w, virtual_screen_h;

// CONFIGURATION variables
int allow_self_connections = 0; // DO NOT ALLOW SELF CONNECTIONS
const float SPIKE_VOLTAGE = 30.0f;
int min_connections = MIN_NEURON_INPUTS, max_connections = MAX_NEURON_INPUTS;

void fill_virtual_screen_emptiness()
{
    for (size_t i = 0; i < (virtual_screen_h * virtual_screen_w); i++)
    {
        live_spike_array_cpu[i] = 128; // Gray
    }
}

int init_network(int input_count, int main_neuron_count, int output_count)
{
    // !!! Function does not require best performance (no real-time usage) !!!
    if (main_neuron_count < output_count)
    {
        print_error("Not enough processing neurons for setting them as outputs. ");
        printf("Asked for: %d processing neurons and %d outputs\n", main_neuron_count, output_count);
        return 1;
    }
    recommended_inhibitory_neuron_count = (int)(main_neuron_count / EX_IN_RATIO); // (int) to prevent data loss warning
    recommended_excitatory_neuron_count = main_neuron_count - recommended_inhibitory_neuron_count;
    if (recommended_excitatory_neuron_count < output_count)
    {
        print_warning("Current neural network configuration does not allign with recommendations. ");
        printf("Output count %d exceeds recommended count of excitatory neurons %d. This means you will have to rely on custom configuration or use inhibitory neurons as outputs.\n", recommended_excitatory_neuron_count, output_count);
    }
    neurons = (struct Neuron *)calloc((size_t)input_count + (size_t)main_neuron_count, sizeof(struct Neuron)); // Calloc to clear any data
    input_neurons = input_count;
    main_neuron_spaces = main_neuron_count;
    output_neurons = output_count;
    virtual_screen_w = (int)ceil(sqrt(input_count + main_neuron_count));
    virtual_screen_h = virtual_screen_w;
    live_spike_array_cpu = calloc(virtual_screen_h * virtual_screen_w, sizeof(char));
    first_output_neuron_index = input_count + main_neuron_count - output_count;
    fill_virtual_screen_emptiness();
    print_success("Network initialized on CPU! ");
    printf("Inputs: %d. Process: %d. Outputs: %d\n", input_neurons, main_neuron_count, output_neurons);
}

int add_neuron(float v, float a, float b, float c, float d)
{
    // !!! Function does not require best performance (no real-time usage) !!!
    if (main_neuron_count >= main_neuron_spaces)
    {
        print_error("Cannot add new neuron because there is no space left.");
        printf("Used all %d main neuron spaces\n", main_neuron_spaces);
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
    // Connector assumes index start from 1 due to index 0 in neuron metadata being no connection.
    from--;
    to--;
    if (from >= main_neuron_count + input_neurons || to >= main_neuron_count + input_neurons || from < 0 || to < 0)
    {
        print_error("Cannot add neuron connection due to invalid neuron index: ");
        failure = 1;
    }
    else if (from == to && !allow_self_connections)
    {
        print_error("Cannot add neuron connection to itself: ");
        failure = 1;
    }
    else if (latency < 1 || latency >= MAX_NEURON_LATENCY)
    {
        print_error("Cannot add neuron connection due to invalid latency: ");
        failure = 1;
    }
    else if (weight < -1 || weight > 0.5) // TODO: Get these values from property file MIN/MAX
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
    size_t at_index = neurons[to].incomming_connections++;
    if (at_index >= MAX_NEURON_INPUTS)
    {
        print_error("Cannot add neuron connection due to full input list: ");
        printf("from: %d, to: %d, latency: %d, weight: %f\n", from, to, latency, weight);
        return 1;
    }
    neurons[to].inputs[at_index] = from + 1;
    neurons[to].latencies[at_index] = latency;
    neurons[to].weights[at_index] = weight;
    return 0;
}

int populate_neuron_network_automatically()
{
    start_chronometer();
    int fallback_mode = 0;
    if (recommended_excitatory_neuron_count + recommended_inhibitory_neuron_count > main_neuron_spaces)
    {
        print_error("Automatic neuron network populating function encountered bad status. ");
        printf("Recommended Ex_n count %d and In_n count %d exceeds total space %d\n", recommended_excitatory_neuron_count, recommended_inhibitory_neuron_count, main_neuron_spaces);
        fallback_mode = 0;
    }
    if (main_neuron_count >= main_neuron_spaces)
    {
        print_error("Cannot automatically populate neural network because it is full. ");
        printf("All %d main neuron spaces used\n", main_neuron_spaces);
        return 1;
    }
    if (fallback_mode)
    {
        // Fill network only with excitatrory neurons due to missconfiguration
        for (size_t i = 0; i < main_neuron_spaces; i++)
        {
            add_neuron(default_membrane_potential_value(), excitatory_a_value(), excitatory_b_value(), excitatory_c_value(), excitatory_d_value());
        }
    }
    else
    {
        for (size_t i = 0; i < recommended_inhibitory_neuron_count; i++)
        {
            add_neuron(default_membrane_potential_value(), inhibitory_a_value(), inhibitory_b_value(), inhibitory_c_value(), inhibitory_d_value());
        }
        for (size_t i = 0; i < recommended_excitatory_neuron_count; i++)
        {
            add_neuron(default_membrane_potential_value(), excitatory_a_value(), excitatory_b_value(), excitatory_c_value(), excitatory_d_value());
        }
    }
    double time_passed = stop_chronometer();
    print_success("Neuron network automatically populated. ");
    printf("Took %.2f miliseconds\n", time_passed);
    return 0;
}

int connect_neuron_network_automatically()
{
    start_chronometer();
    int neuron_connection_count_blueprint = 0, connection_from = 0, latency = 1;
    float connecting_weight = 0;
    for (size_t i = input_neurons; i < input_neurons + main_neuron_count; i++)
    {
        if (i % 20000 == 0 && main_neuron_count > 20000)
        {
            print_info("Neuron network auto-connect progress: ");
            printf("%.2f%%\n", (float)i / (float)main_neuron_count * 100.0f);
        }
        neuron_connection_count_blueprint = get_random_number() * (max_connections - min_connections) + min_connections;
        if (neuron_connection_count_blueprint > MAX_NEURON_INPUTS)
        {
            print_error("Fatal failure while auto-generating neuron connection count. ");
            printf("Tried to make %d connections yet total free spaces %d\n", neuron_connection_count_blueprint, MAX_NEURON_INPUTS);
            return 1;
        }
        for (size_t j = 0; j < neuron_connection_count_blueprint; j++)
        {
            connection_from = get_random_number() * (input_neurons + main_neuron_count - 1);
            if (i == connection_from)
            {
                j--;
                continue;
            }
            // TODO: Set latency based on neuron distance
            latency = 1;
            // TODO: Set weight based on connecting neuron type (check if correct ratios)
            if (connection_from < input_neurons + recommended_inhibitory_neuron_count && connection_from >= input_neurons)
            {
                connecting_weight = get_random_number() * -0.9f - 0.05f;
            }
            else
            {
                connecting_weight = get_random_number() * 0.4f + 0.05f;
            }
            if (add_connection(connection_from + 1, i + 1, latency, connecting_weight))
            {
                // Adding connection failed. Try again
                j--;
                // TODO: PREVENT FOREVER LOOP if too much failures!!!
            }
        }
    }
    double time_passed = stop_chronometer();
    print_success("Neuron network automatically connected. ");
    printf("Took %.2f miliseconds\n", time_passed);
    return 0;
}

float get_step_performance(unsigned int iterations)
{
    // Make sure to init GPU side network after this again because we need clean data.
    float time_passed = 0;
    int gpu_status = 0;
    start_chronometer();
    for (size_t i = 0; i < iterations; i++)
    {
        gpu_status |= simulate_gpu_step();
    }
    time_passed = stop_chronometer()/(double)iterations;
    if (gpu_status)
    {
        time_passed = -1; // Time is invalid because GPU process failed.
    }
    return time_passed;
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
