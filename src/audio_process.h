#pragma once
#include <portaudio.h> // PortAudio

struct paTestData
{
    float left_phase;
    float right_phase;
};

void check_portaudio_error(PaError err);

int patestCallback(const void *inputBuffer, void *outputBuffer,
                   unsigned long framesPerBuffer,
                   const PaStreamCallbackTimeInfo *timeInfo,
                   PaStreamCallbackFlags statusFlags,
                   void *userData);