#include <stdio.h>
#include <string.h> // strcmp
#include "../src/network_cpu.h"
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
    if (total_neuron_spaces != 0)
    {
        print_error_unformatted("Total neuron spaces is not 0! Actual: ");
        printf("%d\n", total_neuron_spaces);
        return 1;
    }
    if (neuron_count != 0)
    {
        print_error_unformatted("Neuron count is not 0! Actual: ");
        printf("%d\n", neuron_count);
        return 1;
    }
    init_network(1000000);
    print_info_unformatted("After initialization!\n");
    if (neurons == NULL)
    {
        print_error_unformatted("Neuron list is NULL!\n");
        return 1;
    }
    if (total_neuron_spaces != 1000000)
    {
        print_error_unformatted("Total neuron spaces is not 10000! Actual: ");
        printf("%d\n", total_neuron_spaces);
        return 1;
    }
    if (neuron_count != 0)
    {
        print_error_unformatted("Neuron count is not 0! Actual: ");
        printf("%d\n", neuron_count);
        return 1;
    }
    free_network();
    print_info_unformatted("After deleting network!\n");
    if (neurons != NULL)
    {
        print_error_unformatted("Neuron list is not NULL!\n");
        return 1;
    }
    if (total_neuron_spaces != 0)
    {
        print_error_unformatted("Total neuron spaces is not 0! Actual: ");
        printf("%d\n", total_neuron_spaces);
        return 1;
    }
    print_success_unformatted("Network initialization test passed!\n");
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
    }
    return failed_tests;
}