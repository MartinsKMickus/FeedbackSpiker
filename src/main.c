#include <stdio.h>     // printf
#include <portaudio.h> // PortAudio
#include <string.h>    // strcmp
#include "audio_process.h"
#include "spiker_network.h"
#include "spiker_network_gpu.cuh"
#include "utilities/text_formatter.h"
#include <time.h> // clock_gettime

#ifdef VERSION
char *APP_VERSION = VERSION;
#else
char *APP_VERSION = "UNDEFINED!";
#endif

#define SAMPLE_RATE (44100)

void diagnostics()
{
    size_t neuron_size = sizeof(struct Neuron);
    size_t diagnostic_neuron_count = 100000;
    print_info("Size of Neuron: ");
    printf("%ld bytes!\n", neuron_size);
    print_info("Trying to initialize network with ");
    printf("%ld neurons! ", diagnostic_neuron_count);
    printf("Network size: %ld bytes, %ld megabytes\n", neuron_size * diagnostic_neuron_count, neuron_size * diagnostic_neuron_count / 1024 / 1024);
    print_info("Initializing network on CPU!\n");
    init_network(diagnostic_neuron_count);
    for (size_t i = 0; i < diagnostic_neuron_count; i++)
    {
        add_neuron(0, 0, 0, 0, 0);
    }
    print_info("Initializing network on GPU!\n");
    init_gpu_network();
    struct timespec start, end;
    print_info("Simulating 100000 steps on GPU!\n");
    clock_gettime(CLOCK_MONOTONIC, &start);
    for (size_t i = 0; i < 100000; i++)
    {
        simulate_gpu_step();
    }
    clock_gettime(CLOCK_MONOTONIC, &end);
    long seconds = end.tv_sec - start.tv_sec;
    long nanoseconds = end.tv_nsec - start.tv_nsec;
    double elapsed = seconds + nanoseconds * 1e-9;
    print_info("Execution time on GPU: ");
    printf("%.9f seconds\n", elapsed);
    free_gpu_network();
    free_network();
    // Perform diagnostics-related tasks here
}

int main(int argc, char *argv[])
{
    // Check for command line arguments
    for (int i = 1; i < argc; i++) {
        if (strcmp(argv[i], "--diagnostics") == 0) {
            print_info("Diagnostics mode!\n");
            diagnostics();
            return 0;
        }
        else if (strcmp(argv[i], "--version") == 0) {
            print_info("Feedback Spiker! Version: ");
            printf("%s\n", APP_VERSION);
            return 0;
        }
        else if (strcmp(argv[i], "--help") == 0) {
            printf("Feedback Spiker! Version: %s\n", APP_VERSION);
            printf("Usage: feedback_spiker [OPTION]\n");
            printf("Options:\n");
            printf("  --diagnostics  Perform diagnostics\n");
            printf("  --version      Show version information\n");
            printf("  --help         Show this help message\n");
            return 0;
        }
    }

    printf("Feedback Spiker! Version: %s\n", APP_VERSION);
    PaError err;
    err = Pa_Initialize();
    check_portaudio_error(err);

    struct paTestData data;

    PaStream *stream;
    /* Open an audio I/O stream. */
    err = Pa_OpenDefaultStream(&stream,
                               1,         /* no input channels */
                               1,        /* stereo output */
                               paFloat32, /* 32 bit floating point output */
                               SAMPLE_RATE,
                               128,            /* frames per buffer, i.e. the number
                                                      of sample frames that PortAudio will
                                                      request from the callback. Many apps
                                                      may want to use
                                                      paFramesPerBufferUnspecified, which
                                                      tells PortAudio to pick the best,
                                                      possibly changing, buffer size.*/
                               patestCallback, /* this is your callback function */
                               &data);         /*This is a pointer that will be passed to
                                                         your callback*/
    check_portaudio_error(err);
    printf("====================================\n");
    Pa_Sleep(1000);

    err = Pa_StartStream(stream);
    check_portaudio_error(err);
    getchar();

    // Pa_Sleep(15000);

    err = Pa_StopStream( stream );
    check_portaudio_error(err);

    err = Pa_Terminate();
    check_portaudio_error(err);

    return 0;
}