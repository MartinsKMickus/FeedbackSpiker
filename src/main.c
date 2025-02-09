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
#include "screen_utils/array_transformer.h"


#ifdef _WIN32
#include "win_screen/screen_framework.h"
//#include <windows.h>
#endif


#ifdef VERSION
char APP_VERSION[] = VERSION;
#else
char APP_VERSION[] = "UNDEFINED!";
#endif


void demo_feedback(struct AudioRingBuffer* buf)
{
    size_t neuron_size = sizeof(struct Neuron);
    int diagnostic_neuron_count = 32768;
    init_screen();
    init_dest_screen(g_Width, g_Height);
    init_network(AUDIO_FRAMES_TO_PROCESS * sizeof(AUDIO_RESOLUTION_TYPE) * 8, diagnostic_neuron_count, AUDIO_FRAMES_TO_PROCESS * sizeof(AUDIO_RESOLUTION_TYPE) * 8);
    populate_neuron_network_automatically();
    connect_neuron_network_automatically();
    init_gpu_network();
    print_info("Measuring GPU step performance!\n");
    double time_passed = get_step_performance(1);
    print_info("One step processing speed on GPU is: ");
    printf("%.2f miliseconds\n", time_passed);
    print_info("Screen information: ");
    printf("Display: %d X %d. Virtual: %d X %d\n", g_Width, g_Height, virtual_screen_w, virtual_screen_h);
    while (1)
    {
        // Check if audio caller is in front of us
        if ((buf->caller - buf->processor + RING_BUFFER_SIZE) % RING_BUFFER_SIZE < AUDIO_FRAMES_TO_PROCESS)
        {
            // Not enough data gathered
            continue;
        }
        start_chronometer();
        for (size_t i = 0; i < AUDIO_FRAMES_TO_PROCESS * sizeof(AUDIO_RESOLUTION_TYPE) * 8;i+=8)
        {
            neurons[i].spike_train <<= 1;
            neurons[i + 1].spike_train <<= 1;
            neurons[i + 2].spike_train <<= 1;
            neurons[i + 3].spike_train <<= 1;
            neurons[i + 4].spike_train <<= 1;
            neurons[i + 5].spike_train <<= 1;
            neurons[i + 6].spike_train <<= 1;
            neurons[i + 7].spike_train <<= 1;
            neurons[i].spike_train |= buf->buffer_ring[buf->processor] & 1 << 0 && 1;
            neurons[i + 1].spike_train |= buf->buffer_ring[buf->processor] & 1 << 1 && 1;
            neurons[i + 2].spike_train |= buf->buffer_ring[buf->processor] & 1 << 2 && 1;
            neurons[i + 3].spike_train |= buf->buffer_ring[buf->processor] & 1 << 3 && 1;
            neurons[i + 4].spike_train |= buf->buffer_ring[buf->processor] & 1 << 4 && 1;
            neurons[i + 5].spike_train |= buf->buffer_ring[buf->processor] & 1 << 5 && 1;
            neurons[i + 6].spike_train |= buf->buffer_ring[buf->processor] & 1 << 6 && 1;
            neurons[i + 7].spike_train |= buf->buffer_ring[buf->processor] & 1 << 7 && 1;
            buf->buffer_ring[buf->processor] = (live_spike_array_cpu[first_output_neuron_index + i] & 1 << 0) << 0;
            buf->buffer_ring[buf->processor] |= (live_spike_array_cpu[first_output_neuron_index + i + 1] & 1 << 0) << 1;
            buf->buffer_ring[buf->processor] |= (live_spike_array_cpu[first_output_neuron_index + i + 2] & 1 << 0) << 2;
            buf->buffer_ring[buf->processor] |= (live_spike_array_cpu[first_output_neuron_index + i + 3] & 1 << 0) << 3;
            buf->buffer_ring[buf->processor] |= (live_spike_array_cpu[first_output_neuron_index + i + 4] & 1 << 0) << 4;
            buf->buffer_ring[buf->processor] |= (live_spike_array_cpu[first_output_neuron_index + i + 5] & 1 << 0) << 5;
            buf->buffer_ring[buf->processor] |= (live_spike_array_cpu[first_output_neuron_index + i + 6] & 1 << 0) << 6;
            buf->buffer_ring[buf->processor++] |= (live_spike_array_cpu[first_output_neuron_index + i + 7] & 1 << 0) << 7;
            if (buf->processor >= RING_BUFFER_SIZE)
            {
                buf->processor = 0;
            }
        }
        refresh_gpu_inputs_from_cpu();
        simulate_gpu_step();
        transfer_gpu_spike_array_to_cpu();
        resize_2d_array_nearest(live_spike_array_cpu, virtual_screen_w, virtual_screen_h, destination_screen, g_Width, g_Height);
        fill_white_pixels(destination_screen);
        double elapsed = stop_chronometer();
        if (elapsed > process_fill_time)
        {
            print_error("Process is too slow and cannot process audio in real time. Iteration: ");
            printf("%.2f miliseconds\n", elapsed);
        }
    }
}

void audio_loopback(struct AudioRingBuffer *buf)
{
    
    while (1)
    {
        // Check if audio caller is in front of us
        if ((buf->caller - buf->processor + RING_BUFFER_SIZE) % RING_BUFFER_SIZE < AUDIO_FRAMES_TO_PROCESS)
        {
            // Not enough data gathered
            Sleep(process_fill_time);
            continue;
        }
        for (int i = 0; i < AUDIO_FRAMES_TO_PROCESS; i++)
        {
            buf->buffer_ring[buf->processor] = buf->buffer_ring[buf->processor++];
            if (buf->processor >= RING_BUFFER_SIZE)
            {
                buf->processor = 0;
            }
        }
    }
}

static void diagnostics()
{
    size_t neuron_size = sizeof(struct Neuron);
    int diagnostic_neuron_count = 1000;
    init_screen();
    init_dest_screen(g_Width, g_Height);
    print_info("Size of Neuron: ");
    printf("%llu bytes!\n", neuron_size);
    print_info("Trying to initialize network with ");
    printf("%d neurons! ", diagnostic_neuron_count);
    printf("Network size: %llu bytes, %llu megabytes\n", neuron_size * diagnostic_neuron_count, neuron_size * diagnostic_neuron_count / 1024 / 1024);
    print_info("Initializing network on CPU!\n");
    init_network(10, diagnostic_neuron_count, 0);
    print_info("Populating network!\n");
    populate_neuron_network_automatically();
    print_info("Connecting network!\n");
    connect_neuron_network_automatically();
    print_info("Initializing network on GPU!\n");
    init_gpu_network();
    print_info("Measuring GPU step performance!\n");
    double time_passed = get_step_performance(50);
    print_info("One step processing speed on GPU is: ");
    printf("%.2f miliseconds\n", time_passed);
    print_info("Simulating 10000 steps on GPU!\n");
    start_chronometer();
    for (size_t i = 0; i < 10000; i++)
    {
        neurons[0].spike_train = 1 << 0;
        neurons[1].spike_train = 1 << 0;
        neurons[2].spike_train = 1 << 0;
        neurons[3].spike_train = 1 << 0;
        neurons[4].spike_train = 1 << 0; 
        neurons[5].spike_train = 1 << 0;
        neurons[6].spike_train = 1 << 0;
        //neurons[7].spike_train = 1 << 0;
        neurons[8].spike_train = 1 << 0;
        refresh_gpu_inputs_from_cpu();
        simulate_gpu_step();
        transfer_gpu_spike_array_to_cpu();
        resize_2d_array_nearest(live_spike_array_cpu, virtual_screen_w, virtual_screen_h, destination_screen, g_Width, g_Height);
        fill_white_pixels(destination_screen);
        //Sleep(1);
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
        if (strcmp(argv[i], "--help") == 0)
        {
            printf("Feedback Spiker! Version: %s\n", APP_VERSION);
            printf("Usage: feedback_spiker [OPTION]\n");
            printf("Options:\n");
            printf("  --diagnostics  Perform diagnostics\n");
            printf("  --version      Show version information\n");
            printf("  --help         Show this help message\n");
            return 0;
        }
        else if (strcmp(argv[i], "--diagnostics") == 0)
        {
            print_info("Diagnostics mode!\n");
            diagnostics();
            return 0;
        }
        else if (strcmp(argv[i], "--version") == 0)
        {
            print_info("Feedback Spiker! Version: ");
            printf("%s\n", APP_VERSION);
            return 0;
        }
        else if (strcmp(argv[i], "--demo"))
        {
            print_info("Demonstration mode!\n");
        }
    }
   /* print_info("No arguments provided. Entering diagnostics mode!\n");
    diagnostics();
    return 0;*/
    PaError err;

    printf("Feedback Spiker! Version: %s\n", APP_VERSION);
    err = Pa_Initialize();
    check_portaudio_error(err);

    struct AudioRingBuffer data = { 0 };
    data.caller = 0;
    data.processor = 0;
    PaStream *stream;
    /* Open an audio I/O stream. */
    err = Pa_OpenDefaultStream(&stream,
                              1,         /* no input channels */
                              1,        /* stereo output */
                              paFloat32, /* 32 bit floating point output */
                              SAMPLE_RATE,
                              AUDIO_FRAMES_TO_PROCESS,            /* frames per buffer, i.e. the number
                                                     of sample frames that PortAudio will
                                                     request from the callback. Many apps
                                                     may want to use
                                                     paFramesPerBufferUnspecified, which
                                                     tells PortAudio to pick the best,
                                                     possibly changing, buffer size.*/
                              audio_process_callback, /* this is your callback function */
                              &data);         /*This is a pointer that will be passed to
                                                        your callback*/
    check_portaudio_error(err);
    printf("====================================\n");
    print_warning("Remember to random seed! Currently fixed 1\n");
    Pa_Sleep(2000);
    print_info("Audio buffer reset time: ");
    printf("%.3f milliseconds\n", buffer_reset_time);
    print_info("Audio buffer process fill time: ");
    printf("%.3f milliseconds\n", process_fill_time);

    err = Pa_StartStream(stream);
    check_portaudio_error(err);
    Pa_Sleep(2000);

    //audio_loopback(&data);
    demo_feedback(&data);

    // Pa_Sleep(15000);

    err = Pa_StopStream( stream );
    check_portaudio_error(err);

    err = Pa_Terminate();
    check_portaudio_error(err);

    return 0;
}