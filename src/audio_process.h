#pragma once
#include <portaudio.h> // PortAudio

#define SAMPLE_RATE 48000
#define AUDIO_FRAMES_TO_PROCESS 512
#define RING_BUFFER_SIZE 2048
#define AUDIO_RESOLUTION_TYPE int
#define AUDIO_RESOLUTION_TYPE_CAPACITY 4294967296

/// <summary>
/// Buffer reset time shows when old data will be started to be overwritten.
/// </summary>
extern float buffer_reset_time;
/// <summary>
/// Process fill time is minimum time for a process unit to wait before gathering data from audio buffer.
/// After this time one full data array is filled that can be processed.
/// </summary>
extern float process_fill_time;

struct AudioRingBuffer {
    AUDIO_RESOLUTION_TYPE buffer_ring[RING_BUFFER_SIZE];
    int caller, processor;
};

void check_portaudio_error(PaError err);

// To be called in float mode!!!
int audio_process_callback(const void *inputBuffer, void *outputBuffer,
                   unsigned long framesPerBuffer,
                   const PaStreamCallbackTimeInfo *timeInfo,
                   PaStreamCallbackFlags statusFlags,
                   void *userData);