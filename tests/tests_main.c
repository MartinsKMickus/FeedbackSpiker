#include <stdio.h>
#include <string.h> // strcmp
#include "../src/spiker_network.h"
#include "../src/utilities/text_formatter.h"

int test_network_initialization()
{
    print_info_unformatted("Testing network initialization!\n");
    print_info_unformatted("Before initialization!\n");
    if (neurons != NULL)
    {
        print_error_unformatted("Neuron list is not NULL!\n");
        return 1;
    }
    if (main_neuron_spaces != 0)
    {
        print_error_unformatted("Total neuron spaces is not 0! Actual: ");
        printf("%d\n", main_neuron_spaces);
        return 1;
    }
    if (main_neuron_count != 0)
    {
        print_error_unformatted("Neuron count is not 0! Actual: ");
        printf("%d\n", main_neuron_count);
        return 1;
    }
    init_network(0, 1000000, 0);
    print_info_unformatted("After initialization!\n");
    if (neurons == NULL)
    {
        print_error_unformatted("Neuron list is NULL!\n");
        return 1;
    }
    if (main_neuron_spaces != 1000000)
    {
        print_error_unformatted("Total neuron spaces is not 10000! Actual: ");
        printf("%d\n", main_neuron_spaces);
        return 1;
    }
    if (main_neuron_count != 0)
    {
        print_error_unformatted("Neuron count is not 0! Actual: ");
        printf("%d\n", main_neuron_count);
        return 1;
    }
    free_network();
    print_info_unformatted("After deleting network!\n");
    if (neurons != NULL)
    {
        print_error_unformatted("Neuron list is not NULL!\n");
        return 1;
    }
    if (main_neuron_spaces != 0)
    {
        print_error_unformatted("Total neuron spaces is not 0! Actual: ");
        printf("%d\n", main_neuron_spaces);
        return 1;
    }
    print_success_unformatted("Network initialization test passed!\n");
    return 0;
}

int test_add_neuron()
{
    print_info_unformatted("Testing add neuron!\n");
    init_network(0, 10, 0);
    if (main_neuron_count != 0)
    {
        print_error_unformatted("Neuron count is not 0! Actual: ");
        printf("%d\n", main_neuron_count);
        return 1;
    }
    add_neuron(-0.65f, 0.02f, 0.2f, -65.0f, 8.0f);
    print_info_unformatted("After adding first neuron!\n");
    if (main_neuron_count != 1)
    {
        print_error_unformatted("Neuron count is not 1! Actual: ");
        printf("%d\n", main_neuron_count);
        return 1;
    }
    add_neuron(-0.65f, 0.02f, 0.2f, -65.0f, 8.0f);
    print_info_unformatted("After adding second neuron!\n");
    if (main_neuron_count != 2)
    {
        print_error_unformatted("Neuron count is not 2! Actual: ");
        printf("%d\n", main_neuron_count);
        return 1;
    }
    free_network();
    print_success_unformatted("Add neuron test passed!\n");
    return 0;
}

int main(int argc, char **argv)
{
    printf("Test executable launched!\n");
    unsigned int failed_tests = 0;
    // Check for command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "network_initialization") == 0) {
            failed_tests += test_network_initialization();
        }
        else if (strcmp(argv[i], "add_neuron") == 0) {
            failed_tests += test_add_neuron();
        }
    }
    return failed_tests;
}