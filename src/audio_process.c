#include <portaudio.h> // PortAudio
#include "audio_process.h"
#include <stdio.h>  // printf
#include <stdlib.h> // exit

float buffer_reset_time = 1000.0f * (RING_BUFFER_SIZE - AUDIO_FRAMES_TO_PROCESS) / SAMPLE_RATE;
float process_fill_time = 1000.0f * AUDIO_FRAMES_TO_PROCESS / SAMPLE_RATE;

void check_portaudio_error(PaError err)
{
    if (err != paNoError)
    {
        printf("PortAudio error: %s\n", Pa_GetErrorText(err));
        Pa_Terminate();
        exit(1);
    }
}

int audio_process_callback(const void *inputBuffer, void *outputBuffer,
                   unsigned long framesPerBuffer,
                   const PaStreamCallbackTimeInfo *timeInfo,
                   PaStreamCallbackFlags statusFlags,
                   void *userData)
{
    /* Cast data passed through stream to our structure. */
    struct AudioRingBuffer*data = (struct AudioRingBuffer*)userData;
    float *out = (float *)outputBuffer;
    unsigned int i;
    float *in = (float *)inputBuffer;
    for (i = 0; i < framesPerBuffer; i++)
    {
        out[i] = (float)data->buffer_ring[data->caller] / (float)AUDIO_RESOLUTION_TYPE_CAPACITY;
        data->buffer_ring[data->caller++] = (int)(in[i] * (float)AUDIO_RESOLUTION_TYPE_CAPACITY);
        if (data->caller >= RING_BUFFER_SIZE)
        {
            data->caller = 0;
        }
    }
    return 0;
}