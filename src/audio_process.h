#pragma once
#include <portaudio.h> // PortAudio

#define SAMPLE_RATE 48000
#define FRAMES_PER_BUFFER 128
#define AUDIO_RESOLUTION_TYPE unsigned int
#define AUDIO_RESOLUTION_TYPE_CAPACITY 4294967296

extern float buffer_reset_time;

struct RotatingDoubleBuffer {
    AUDIO_RESOLUTION_TYPE input_buffer1[FRAMES_PER_BUFFER];
    AUDIO_RESOLUTION_TYPE input_buffer2[FRAMES_PER_BUFFER];
    AUDIO_RESOLUTION_TYPE output_buffer1[FRAMES_PER_BUFFER];
    AUDIO_RESOLUTION_TYPE output_buffer2[FRAMES_PER_BUFFER];
    char input1_writing_ready, input1_reading_ready;
    char input2_writing_ready, input2_reading_ready;
    char output1_writing_ready, output1_reading_ready;
    char output2_writing_ready, output2_reading_ready;
    char ready;
};

void check_portaudio_error(PaError err);

int audio_process_callback(const void *inputBuffer, void *outputBuffer,
                   unsigned long framesPerBuffer,
                   const PaStreamCallbackTimeInfo *timeInfo,
                   PaStreamCallbackFlags statusFlags,
                   void *userData);

void initialize_audio_buffer(struct RotatingDoubleBuffer * buf);