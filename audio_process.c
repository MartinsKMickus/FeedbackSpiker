#include <portaudio.h> // PortAudio
#include "audio_process.h"
#include <stdio.h>  // printf
#include <stdlib.h> // exit

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
    struct paTestData *data = (struct paTestData *)userData;
    float *out = (float *)outputBuffer;
    unsigned int i;
    float *in = (float *)inputBuffer;

    for (i = 0; i < framesPerBuffer; i++)
    {
        // out[i] = data->left_phase;  /* left */
        // out[i] = data->right_phase; /* right */
        out[i] = in[i]; /* left */
        // out[i + 1] = in[i + 1]; /* right */
        // /* Generate simple sawtooth phaser that ranges between -1.0 and 1.0. */
        // data->left_phase += 0.01f;
        // /* When signal reaches top, drop back down. */
        // if (data->left_phase >= 1.0f)
        //     data->left_phase -= 2.0f;
        // /* higher pitch so we can distinguish left and right. */
        // data->right_phase += 0.03f;
        // if (data->right_phase >= 1.0f)
        //     data->right_phase -= 2.0f;
    }
    return 0;
}