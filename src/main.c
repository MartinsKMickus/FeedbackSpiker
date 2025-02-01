#include <stdio.h>     // printf
#include <portaudio.h> // PortAudio (still haven't got rid of this :D)
#include <string.h>    // strcmp
#include "audio_process.h"
#include "spiker_network.h"
#include "spiker_network_gpu.h"
#include "utilities/text_formatter.h"
#include <time.h> // clock_gettime
#include "utilities/input_handler.h"
#include "utilities/code_measurements.h"
#include "neuron_properties.h"


#ifdef VERSION
char APP_VERSION[] = VERSION;
#else
char APP_VERSION[] = "UNDEFINED!";
#endif

static void diagnostics()
{
    size_t neuron_size = sizeof(struct Neuron);
    int diagnostic_neuron_count = 100000;
    print_info("Size of Neuron: ");
    printf("%llu bytes!\n", neuron_size);
    print_info("Trying to initialize network with ");
    printf("%d neurons! ", diagnostic_neuron_count);
    printf("Network size: %llu bytes, %llu megabytes\n", neuron_size * diagnostic_neuron_count, neuron_size * diagnostic_neuron_count / 1024 / 1024);
    print_info("Initializing network on CPU!\n");
    init_network(0, diagnostic_neuron_count, 0);
    print_info("Populating network!\n");
    populate_neuron_network_automatically();
    print_info("Connecting network!\n");
    connect_neuron_network_automatically();
    print_info("Initializing network on GPU!\n");
    init_gpu_network();
    print_info("Simulating 100000 steps on GPU!\n");
    start_chronometer();
    for (size_t i = 0; i < 100000; i++)
    {
        simulate_gpu_step();
    }
    double elapsed = stop_chronometer();
    print_info("Execution time on GPU: ");
    printf("%.2f miliseconds\n", elapsed);
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
    print_info("No arguments provided. Entering diagnostics mode!\n");
    diagnostics();
    return 0;
    PaError err;

    printf("Feedback Spiker! Version: %s\n", APP_VERSION);
    err = Pa_Initialize();
    check_portaudio_error(err);

    struct RotatingDoubleBuffer data;
    PaStream *stream;
    /* Open an audio I/O stream. */
    err = Pa_OpenDefaultStream(&stream,
                              1,         /* no input channels */
                              1,        /* stereo output */
                              paFloat32, /* 32 bit floating point output */
                              SAMPLE_RATE,
                              FRAMES_PER_BUFFER,            /* frames per buffer, i.e. the number
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
    Pa_Sleep(2000);
    print_info("Audio buffer reset time: ");
    printf("%.3f milliseconds\n", buffer_reset_time);

    err = Pa_StartStream(stream);
    check_portaudio_error(err);
    Pa_Sleep(2000);

    AUDIO_RESOLUTION_TYPE *donor;
    AUDIO_RESOLUTION_TYPE *receiver;
    char* buffer_mode = &data.current_buffer;

    while (1)
    {
        char local_buffer_mode = *buffer_mode;

        if (*buffer_mode == local_buffer_mode) {
            continue;
        }

        *buffer_mode = local_buffer_mode;

        // Process the active buffer
        donor = (*buffer_mode == 1) ? data.input_buffer1 : data.input_buffer2;
        receiver = (*buffer_mode == 1) ? data.output_buffer1 : data.output_buffer2;

        for (int i = 0; i < FRAMES_PER_BUFFER; i++) {
            receiver[i] = donor[i];
        }

        if (local_buffer_mode != *buffer_mode) {
            print_warning("Active audio buffer changed while processing!\n");
        }
    }

    // Pa_Sleep(15000);

    err = Pa_StopStream( stream );
    check_portaudio_error(err);

    err = Pa_Terminate();
    check_portaudio_error(err);

    return 0;
}