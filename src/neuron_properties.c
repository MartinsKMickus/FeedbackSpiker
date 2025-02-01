#include "neuron_properties.h"
#include <time.h>
#include <stdlib.h>  // Required for rand() and RAND_MAX

int neuron_properties_initialized = 0;
long long neuron_property_seed = 1;

static void init_neuron_properties()
{
	if (neuron_properties_initialized)
	{
		return;
	}
	neuron_property_seed = time(0);
	srand(neuron_property_seed);
	neuron_properties_initialized = 1;
}

float get_random_number()
{
	init_neuron_properties();
	return ((float)rand() / (float)RAND_MAX);
}

float default_membrane_potential_value()
{
	return -65.0f;
}

float excitatory_a_value()
{
	return 0.02f;
}

float excitatory_b_value()
{
	return 0.2f;
}

float excitatory_c_value()
{
	float r = get_random_number();
	return -65.0f + 15.0f * r * r;
}

float excitatory_d_value()
{
	float r = get_random_number();
	return 8.0f - 6.0f * r * r;
}

float inhibitory_a_value()
{
	float r = get_random_number();
	return 0.02f + 0.08f * r;
}

float inhibitory_b_value()
{
	float r = get_random_number();
	return 0.25f - 0.05f * r;
}

float inhibitory_c_value()
{
	return -65.0f;
}

float inhibitory_d_value()
{
	return 2.0f;
}
