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

int audio_process_callback(const void *inputBuffer, void *outputBuffer,
                   unsigned long framesPerBuffer,
                   const PaStreamCallbackTimeInfo *timeInfo,
                   PaStreamCallbackFlags statusFlags,
                   void *userData)
{
    /* Cast data passed through stream to our structure. */
    struct RotatingDoubleBuffer *data = (struct RotatingDoubleBuffer *)userData;
    if (!data->ready)
    {
        return;
    }
    float *out = (float *)outputBuffer;
    unsigned int i;
    float *in = (float *)inputBuffer;
    AUDIO_RESOLUTION_TYPE running_value;
    AUDIO_RESOLUTION_TYPE *current_input;
    AUDIO_RESOLUTION_TYPE *current_output;
    char current_input_buffer, current_output_buffer;
    if (data->input1_writing_ready)
    {
        current_input = data->input_buffer1;
        current_input_buffer = 1;
    }
    else if (data->input2_writing_ready)
    {
        current_input = data->input_buffer2;
        current_input_buffer = 2;
    }
    else
    {
        printf("Fatal audio buffer overflow (recorded data discarded)\n");
        return;
    }
    if (data->output1_reading_ready)
    {
        current_output = data->output_buffer1;
        current_output_buffer = 1;
    }
    else if (data->output2_reading_ready)
    {
        current_output = data->output_buffer2;
        current_output_buffer = 2;
    }
    else
    {
        printf("Fatal audio buffer underflow (data not received for playback)\n");
        return;
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
    // Firstly telling that output is gone.
    if (current_output_buffer == 1)
    {
        data->output1_reading_ready = 0;
        data->output1_writing_ready = 1; // Final action if on, processor will write
    }
    else
    {
        data->output2_reading_ready = 0;
        data->output2_writing_ready = 1; // Final action if on, processor will write
    }
    if (current_input_buffer == 1)
    {
        data->input1_writing_ready = 0;
        data->input1_reading_ready = 1; // Final action if on, processor will read
    }
    else
    {
        data->input2_writing_ready = 0;
        data->input2_reading_ready = 1; // Final action if on, processor will read
    }
    
    return 0;
}

void initialize_audio_buffer(struct RotatingDoubleBuffer *buf)
{
    buf->input1_writing_ready = 1;
    buf->input2_writing_ready = 1;
    buf->output1_reading_ready = 1;
    buf->output2_reading_ready = 1;
    buf->ready = 1;
}