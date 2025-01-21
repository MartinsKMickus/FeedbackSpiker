#include <portaudio.h> // PortAudio
#include "audio_process.h"
#include <stdio.h>  // printf
#include <stdlib.h> // exit

float buffer_reset_time = 1000.0f * FRAMES_PER_BUFFER / SAMPLE_RATE * 2;

void check_portaudio_error(PaError err)
{
    if (err != paNoError)
    {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        Pa_Terminate();
        exit(1);
    }
}

int patestCallback(const void *inputBuffer, void *outputBuffer,
                   unsigned long framesPerBuffer,
                   const PaStreamCallbackTimeInfo *timeInfo,
                   PaStreamCallbackFlags statusFlags,
                   void *userData)
{
    /* Cast data passed through stream to our structure. */
    struct RotatingDoubleBuffer *data = (struct RotatingDoubleBuffer *)userData;
    float *out = (float *)outputBuffer;
    unsigned int i;
    float *in = (float *)inputBuffer;
    AUDIO_RESOLUTION_TYPE running_value;
    AUDIO_RESOLUTION_TYPE *current_input;
    AUDIO_RESOLUTION_TYPE *current_output;
    switch (data->current_buffer)
    {
    case 1:
        current_input = data->input_buffer1;
        current_output = data->output_buffer1;
        break;
    case 2:
        current_input = data->input_buffer2;
        current_output = data->output_buffer2;
        break;
    }

    for (i = 0; i < framesPerBuffer; i++)
    {
        running_value = (in[i]+1.0f)*AUDIO_RESOLUTION_TYPE_CAPACITY/2;
        current_input[i] = running_value;
        // out[i] = (float)running_value/AUDIO_RESOLUTION_TYPE - 1; /* left */
        // out[i] = data->left_phase;  /* left */
        // out[i] = data->right_phase; /* right */
        // printf("Data at: %d\n", j);
        out[i] = (float)running_value*2/AUDIO_RESOLUTION_TYPE_CAPACITY - 1;
        // out[i] = (float)current_output[i]*2/AUDIO_RESOLUTION_TYPE_CAPACITY - 1;1
        // current_output[i * framesPerBuffer + j] = 0;
    }
    data->current_buffer = data->current_buffer % 2 + 1;
    return 0;
}