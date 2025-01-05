#include <stdio.h>     // printf
#include <portaudio.h> // PortAudio
#include "audio_process.h"

#ifdef VERSION
char *APP_VERSION = VERSION;
#else
char *APP_VERSION = "UNDEFINED!";
#endif

#define SAMPLE_RATE (48000)

int main()
{
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